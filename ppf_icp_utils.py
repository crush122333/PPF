# -*- coding: utf-8 -*-
"""
ppf_icp_utils.py
辅助函数：法向估计/一致化、Patch PCA+曲率、候选解析、ICP 循环（含镜像修正）、可视化等
"""

import numpy as np
import open3d as o3d
import cv2 as cv

_EPS = 1e-12


# ---------- 法向估计 / 一致化 ----------
def estimate_normals_consistent_knn(pcd: o3d.geometry.PointCloud, k: int) -> o3d.geometry.PointCloud:
    """k 近邻 PCA 法向 + 统一朝向（相对全局中心的外侧），与 MATLAB 等价"""
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(3, int(k))))         #计算每个领域点的法向 kdtree领域检索
    xyz = np.asarray(pcd.points)
    ctr = xyz.mean(axis=0)                                                                           #计算均值中心点，就是点云的质心 axis=0对列取平均，就是每个坐标（x,y,z）
    nrm = np.asarray(pcd.normals)
    v   = xyz - ctr                                                                                  #点云三维坐标指向质心的向量
    mask = (np.sum(nrm * v, axis=1) > 0)                                                             #判断法向相反的点
    nrm[mask] *= -1.0                                                                                #法向进行翻转
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    return pcd

def unify_normals_orientation(model_pcd: o3d.geometry.PointCloud,
                              scene_pcd: o3d.geometry.PointCloud):
    """
    统一模型和场景点云的法向朝向方向（使整体平均法向一致）
    原理：比较平均法向方向，如果夹角>90°则翻转其中之一
    """
    n_model = np.mean(np.asarray(model_pcd.normals), axis=0)
    n_scene = np.mean(np.asarray(scene_pcd.normals), axis=0)
    n_model /= (np.linalg.norm(n_model) + _EPS)
    n_scene /= (np.linalg.norm(n_scene) + _EPS)

    if np.dot(n_model, n_scene) < 0:  # 如果反向
        scene_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(scene_pcd.normals))
        print("[INFO] flipped scene normals for consistency")

    return model_pcd, scene_pcd

# ---------- Patch PCA + 曲率 ----------
def _curvature_from_cov(P: np.ndarray):
    if P.shape[0] < 3:                                                                              #邻域点数少于 3 个无法做 3D PCA/协方差分解，直接返回空结果
        return None, None, None
    c = P.mean(axis=0)                                                                              #把所有点移到以质心为原点的坐标系（中心化），后面算协方差要用中心化数据
    Q = P - c
    C = (Q.T @ Q) / max(1, P.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    n = vecs[:, 0]
    kappa = vals[0] / max(_EPS, vals.sum())
    return n, kappa, c


def to_ppf_array_patch(pcd: o3d.geometry.PointCloud,
                       radius: float, min_pts: int,
                       curv_min: float, curv_max: float,
                       stride: int, voxel_merge: float) -> np.ndarray:
    """半径邻域 → PCA 法向 → 曲率筛选 → 一致化 →（可选）体素合并；返回 Nx6: [x y z nx ny nz]"""
    xyz_all = np.asarray(pcd.points)
    if xyz_all.size == 0:
        return np.zeros((0, 6), np.float32)
    ctr_all = xyz_all.mean(axis=0)

    kdt = o3d.geometry.KDTreeFlann(pcd)
    keep = []
    step = max(1, int(stride))
    for idx in range(0, len(xyz_all), step):
        p = xyz_all[idx]
        k, nbr_idx, _ = kdt.search_radius_vector_3d(p, float(radius))
        if k < min_pts:
            continue
        Pn = xyz_all[np.asarray(nbr_idx, dtype=int)]
        n, kappa, c = _curvature_from_cov(Pn)
        if n is None:
            continue
        if np.dot(n, (c - ctr_all)) > 0:
            n = -n
        if not (curv_min <= kappa <= curv_max):
            continue
        n = n / max(_EPS, np.linalg.norm(n))
        keep.append(np.hstack([c, n]))

    A = np.array(keep, dtype=np.float32) if keep else np.zeros((0, 6), np.float32)

    # 可选体素合并
    if voxel_merge and len(A) > 0:
        pc_patch = o3d.geometry.PointCloud()
        pc_patch.points  = o3d.utility.Vector3dVector(A[:, :3])
        pc_patch.normals = o3d.utility.Vector3dVector(A[:, 3:])
        pc_patch = pc_patch.voxel_down_sample(voxel_size=float(voxel_merge))
        # voxel_down_sample 可能丢法向：最近邻回填
        if len(pc_patch.normals) != len(pc_patch.points):
            from sklearn.neighbors import NearestNeighbors
            nbr = NearestNeighbors(n_neighbors=1).fit(A[:, :3])
            idx = nbr.kneighbors(np.asarray(pc_patch.points), return_distance=False).ravel()
            nrm = A[idx, 3:]
            pc_patch.normals = o3d.utility.Vector3dVector(nrm)
        A = np.hstack([np.asarray(pc_patch.points), np.asarray(pc_patch.normals)]).astype(np.float32)

    return A


# ---------- PPF 候选解析 / 调试 ----------
def debug_get_props(py_results):
    n = len(py_results)
    print(f"共有 {n} 个候选")
    for i in range(min(n, 5)):
        r = py_results[i]
        if hasattr(r, "pose"):
            P = np.array(r.pose, dtype=np.float64).reshape(4, 4)
            print(f"--- Candidate #{i} ---")
            print("Pose:\n", P)
        for name in ("score", "residual", "similarity", "numVotes", "votes"):
            if hasattr(r, name):
                print(f"{name} =", getattr(r, name))


def extract_candidates(py_results):
    n = len(py_results)                                 #获取候选的个数
    poses, votes, residual = [], [], []                 #创建三个空列表
    for i in range(n):
        r = py_results[i]
        P = np.array(r.pose, dtype=np.float64).reshape(4, 4)  #对象中位姿矩阵，然后转化成4*4的矩阵
        poses.append(P.astype(np.float32))                    #把矩阵 P 转成 float32 精度，存到pose列表
        if hasattr(r, "numVotes"):
            votes.append(int(r.numVotes))
        elif hasattr(r, "votes"):
            votes.append(int(r.votes))
        else:
            votes.append(0)
        if hasattr(r, "residual"):
            residual.append(float(r.residual))
        else:
            residual.append(np.nan)
    return n, poses, np.array(votes, int), np.array(residual, float)


# ---------- ICP & 姿态修正 ----------
def _parse_icp_output(out):
    # === 常见版本: (status:int, residual:float, pose:ndarray) ===
    if isinstance(out, (tuple, list)):
        if len(out) == 3 and isinstance(out[2], np.ndarray):
            pose4x4 = out[2].astype(np.float32).reshape(4, 4)
            residual = float(out[1])
            return pose4x4, residual

        # 兼容部分版本: (residual, pose) 或 (pose, residual)
        if len(out) == 2:
            a, b = out
            if isinstance(a, np.ndarray):
                return a.astype(np.float32).reshape(4, 4), float(b)
            if isinstance(b, np.ndarray):
                return b.astype(np.float32).reshape(4, 4), float(a)

    # === 如果直接是 Pose3D 或 ndarray ===
    if hasattr(out, "pose"):
        pose4x4 = np.array(out.pose, dtype=np.float32).reshape(4, 4)
        return pose4x4, float("nan")

    if isinstance(out, np.ndarray):
        pose4x4 = out.astype(np.float32).reshape(4, 4)
        return pose4x4, float("nan")

    # === 兜底: 其他未知类型 ===
    raise RuntimeError(f"Unexpected ICP output format: {type(out)} -> {out}")



def _median_nn_dist(points: np.ndarray, scene_xyz: np.ndarray) -> float:
    """
    计算每个 points 点到 scene_xyz 的 1-NN 距离的中位数（更稳健），
    使用双向分块方式避免 O(B*N) 内存爆炸。
    """
    import numpy as np

    # ---- 1. 输入检查 ----
    P = np.asarray(points, dtype=np.float32)
    S = np.asarray(scene_xyz, dtype=np.float32)

    if P.size == 0 or S.size == 0:
        return np.inf

    # 去除 NaN / Inf
    P = P[np.all(np.isfinite(P), axis=1)]
    S = S[np.all(np.isfinite(S), axis=1)]
    if P.size == 0 or S.size == 0:
        return np.inf

    # ---- 2. 分块配置 ----
    pb = 2048     # 模型块大小
    sb = 65536    # 场景块大小（每次处理6万多点）
    mins = np.full(P.shape[0], np.inf, dtype=np.float32)

    # ---- 3. 双重分块计算 1-NN 距离 ----
    for i in range(0, P.shape[0], pb):
        Pi = P[i:i+pb]
        min_i = np.full(Pi.shape[0], np.inf, dtype=np.float32)
        for j in range(0, S.shape[0], sb):
            Sj = S[j:j+sb]
            diff = Pi[:, None, :] - Sj[None, :, :]     # (pb, sb, 3)
            d2 = np.einsum('bij,bij->bi', diff, diff)  # (pb, sb)
            min_i = np.minimum(min_i, d2.min(axis=1))  # 每个点的最小距离平方
        mins[i:i+pb] = np.sqrt(min_i)                  # 存回全局最小距离

    # ---- 4. 检查结果有效性 ----
    mins = mins[np.isfinite(mins)]
    if mins.size == 0:
        return np.inf

    # ---- 5. 返回中位数（稳健度量）----
    return float(np.median(mins))


def _try_pose_variants(pose_in: np.ndarray, model_xyz: np.ndarray, scene_xyz: np.ndarray):
    """验证单一姿态的ICP残差（不再尝试镜像翻转）"""
    R0 = pose_in[:3, :3].astype(np.float64, copy=True)
    t0 = pose_in[:3, 3].astype(np.float64, copy=True)

    M = np.asarray(model_xyz, dtype=np.float64)
    S = np.asarray(scene_xyz, dtype=np.float64)

    Mtf = (M @ R0.T) + t0
    med = _median_nn_dist(Mtf, S)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R0.astype(np.float32)
    T[:3, 3] = t0.astype(np.float32)
    return T, float(med)


def run_icp_for_candidates(model_xyz: np.ndarray,
                           scene_xyz: np.ndarray,
                           poses, votes,
                           icp_max_iter=200, icp_tol=1e-5,
                           icp_rej_scale=2.0, icp_levels=5):
    icp = cv.ppf_match_3d.ICP(int(icp_max_iter), float(icp_tol), float(icp_rej_scale), int(icp_levels))
    best_final_pose = np.eye(4, dtype=np.float32)
    best_residual = np.inf                                          # 残差最小
    best_index = -1                                                 # 最佳候选点的索引号
    all_logs = []                                                   # 每个候选点的日志进行打印

    M = model_xyz.astype(np.float32)
    S = scene_xyz.astype(np.float32)

    for i, (P, v) in enumerate(zip(poses, votes)):                 # zip 将 poses 与 votes 配对；enumerate 给出下标 i
        init_pose = P.astype(np.float32)
        R = init_pose[:3, :3]
        t = init_pose[:3, 3]
        M_init = (M @ R.T) + t

        out = icp.registerModelToScene(M_init.astype(np.float32), S.astype(np.float32))
        icp_pose, _ = _parse_icp_output(out)

        cand1 = icp_pose @ init_pose                               # icp*init  先用PPF粗对齐获取位姿，现在就是icp得到微调，icp_pose @ init_pose代表粗匹配的基础上进行微调
        _, res1 = _try_pose_variants(cand1, M, S)
        cand1_fix = cand1
        print(f"  - cand #{i:02d}: votes={v}, icp_res={res1}, use=icp*init"
)
        all_logs.append((i, v, float(res1)))

        if np.isfinite(res1) and res1 < best_residual:
            best_final_pose = cand1_fix
            best_residual = res1
            best_index = i

    print(f"[ICP] best from candidate #{best_index}, residual={best_residual}")
    return best_final_pose, best_residual, best_index, all_logs


# ---------- 可视化/变换 ----------
def transform_cloud(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    Q = o3d.geometry.PointCloud(pcd)  # copy
    Q.transform(T.astype(np.float64))
    return Q


def show_two_clouds(scene_pc: o3d.geometry.PointCloud,
                    model_pc_aligned: o3d.geometry.PointCloud,
                    title="PPF+ICP 最优"):
    scene_pc.paint_uniform_color([0.0, 0.6, 1.0])
    model_pc_aligned.paint_uniform_color([1.0, 0.2, 0.2])
    o3d.visualization.draw_geometries([scene_pc, model_pc_aligned], window_name=title)
