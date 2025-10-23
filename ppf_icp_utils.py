# -*- coding: utf-8 -*-
"""
ppf_icp_utils.py
辅助函数：法向估计/一致化、Patch PCA+曲率、候选解析、ICP 循环（含镜像修正）、可视化等
"""

import numpy as np
import open3d as o3d
import cv2 as cv
from scipy.spatial import cKDTree
from typing import Tuple, Union, Optional

_EPS = 1e-12


# ---------- 法向估计 / 一致化 ----------
def estimate_normals_consistent_knn(pcd: o3d.geometry.PointCloud, k: int) -> o3d.geometry.PointCloud:
    """k 近邻 PCA 法向 + 统一朝向（相对全局中心的外侧"""
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(3, int(k))))         #计算每个领域点的法向 kdtree（二分法）领域检索
    xyz = np.asarray(pcd.points)
    ctr = xyz.mean(axis=0)                                                                           #计算均值中心点，就是点云的质心 axis=0对列取平均，就是每个坐标（x,y,z）
    nrm = np.asarray(pcd.normals)
    v   = xyz - ctr                                                                                  #点云三维坐标指向质心的向量
    mask = (np.sum(nrm * v, axis=1) > 0)                                                             #判断法向相反的点  axis=1表示列方向的求和 就是判断点乘的结果正负
    nrm[mask] *= -1.0                                                                                #法向进行翻转
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    return pcd

def unify_normals_orientation(model_pcd: o3d.geometry.PointCloud,
                              scene_pcd: o3d.geometry.PointCloud):
    """
    统一模型和场景点云的法向朝向方向（使整体平均法向一致）
    原理：比较平均法向方向，如果夹角>90°则翻转其中之一
    """
    n_model = np.mean(np.asarray(model_pcd.normals), axis=0)    #按列对点云的x,y,z三个方向的法向取一个平均 可以看出整体的朝向
    n_scene = np.mean(np.asarray(scene_pcd.normals), axis=0)
    n_model /= (np.linalg.norm(n_model) + _EPS)                 # 把平均法向量归一化（变成单位向量）
    n_scene /= (np.linalg.norm(n_scene) + _EPS)

    if np.dot(n_model, n_scene) < 0:                            # 如果反向。对两个单位向量点乘，同向 >0 ,反向 <0
        scene_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(scene_pcd.normals))
        print("[INFO] flipped scene normals for consistency")

    return model_pcd, scene_pcd

# ---------- Patch PCA + 曲率 ----------
def _curvature_from_cov(P: np.ndarray):
    if P.shape[0] < 3:                                                                              #邻域点数少于 3 个无法做 3D PCA/协方差分解，直接返回空结果
        return None, None, None
    c = P.mean(axis=0)                                                                              #邻居点集的质心
    Q = P - c                                                                                       #把所有点移到以质心为原点的坐标系（中心化），后面算协方差要用中心化数据
    C = (Q.T @ Q) / max(1, P.shape[0] - 1)                                                          #协方差的计算
    vals, vecs = np.linalg.eigh(C)                                                                  # 获取特征向量以及特征值
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]                                                                           # vals 表示特征值    VESC表示特征向量
    n = vecs[:, 0]                                                                                  # 法向量：PCA 中最小特征值的特征向量就是“变化最小”的方向，即局部平面的法向（单位长度）
    kappa = vals[0] / max(_EPS, vals.sum())                                                         # 曲率 k=lmin /(l0+l1+l2)
    return n, kappa, c                                                                              # 单位法向 n、曲率 kappa、邻域质心 c


def to_ppf_array_patch(pcd: o3d.geometry.PointCloud,
                       radius: float, min_pts: int,
                       curv_min: float, curv_max: float,
                       stride: int, voxel_merge: float) -> np.ndarray:
    """半径邻域 → PCA 法向 → 曲率筛选 → 一致化 →（可选）体素合并；返回 Nx6: [x y z nx ny nz]"""
    xyz_all = np.asarray(pcd.points)
    if xyz_all.size == 0:
        return np.zeros((0, 6), np.float32)
    ctr_all = xyz_all.mean(axis=0)

    kdt = o3d.geometry.KDTreeFlann(pcd)                                 # 点云创建KDTree
    keep = []
    step = max(1, int(stride))                                          # 抽样的步长，采样提速
    for idx in range(0, len(xyz_all), step):
        p = xyz_all[idx]
        k, nbr_idx, _ = kdt.search_radius_vector_3d(p, float(radius))   # 半径查询： 找P半径 r内的点邻居
        if k < min_pts:                                                 # 邻居点太少
            continue
        Pn = xyz_all[np.asarray(nbr_idx, dtype=int)]                   # 邻居点的坐标 放在Pn中
        n, kappa, c = _curvature_from_cov(Pn)                          # 对其中领域点进行协方差获取法向量   单位法向 n、曲率 kappa、邻域质心 c
        if n is None:
            continue
        if np.dot(n, (c - ctr_all)) > 0:                               # 当前点与质心点的法向的一致化
            n = -n
        if not (curv_min <= kappa <= curv_max):                        # 设置曲率的阈值
            continue
        n = n / max(_EPS, np.linalg.norm(n))                           # 单位化法向，避免数值不稳定
        keep.append(np.hstack([c, n]))

    A = np.array(keep, dtype=np.float32) if keep else np.zeros((0, 6), np.float32)    #keep 表示的是筛选后的 [(x,y,z,nx,ny,nz)] .

    # 可选体素合并
    if voxel_merge and len(A) > 0:
        pc_patch = o3d.geometry.PointCloud()
        pc_patch.points  = o3d.utility.Vector3dVector(A[:, :3])
        pc_patch.normals = o3d.utility.Vector3dVector(A[:, 3:])
        pc_patch = pc_patch.voxel_down_sample(voxel_size=float(voxel_merge))                       #在进行进一步的体素聚类采样后，体素类的所有点->取了一个几何中心 不会处理法向的数据，因此整体来说 points减少了
        # voxel_down_sample 可能丢法向：最近邻回填
        #""" 在上述进行聚类采样的时候，法向和点云的数量不匹配，因此就需要把新点的索引号去寻找之前的法向
        if len(pc_patch.normals) != len(pc_patch.points):
            from sklearn.neighbors import NearestNeighbors
            nbr = NearestNeighbors(n_neighbors=1).fit(A[:, :3])                                   #这一行在构建一个“旧点云 A” 的 KDTree 模型，用于后面查找每个新点的最近邻
            idx = nbr.kneighbors(np.asarray(pc_patch.points), return_distance=False).ravel()      # 1.计算新点云中每个点在旧点云中最近的邻居索引  2 return_distance=False 表示只返回索引，不返回距离返  3.回结果 idx 的形状是 (M, 1)（二维）
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
    中位数作为匹配的误差
    """
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
    sb = 65536    # 场景块大小（每次处理6万多点）  分块处理是为了减少点之间的运算
    mins = np.full(P.shape[0], np.inf, dtype=np.float32)

    # ---- 3. 双重分块计算 1-NN 距离 ----
    for i in range(0, P.shape[0], pb):
        Pi = P[i:i+pb]                                 # 模型点（pb,3）
        min_i = np.full(Pi.shape[0], np.inf, dtype=np.float32)
        for j in range(0, S.shape[0], sb):
            Sj = S[j:j+sb]                             # 场景点（sb, 3）
            diff = Pi[:, None, :] - Sj[None, :, :]     # (pb, sb, 3)
            d2 = np.einsum('bij,bij->bi', diff, diff)  # (pb, sb)  每个差向量的平方和
            min_i = np.minimum(min_i, d2.min(axis=1))  # 每个点的最小距离平方
        mins[i:i+pb] = np.sqrt(min_i)                  # 存回全局最小距离

    # ---- 4. 检查结果有效性 ----
    mins = mins[np.isfinite(mins)]
    if mins.size == 0:
        return np.inf

    # ---- 5. 返回中位数（稳健度量）----
    return float(np.median(mins))


def _try_pose_variants(pose_in: np.ndarray, model_xyz: np.ndarray, scene_xyz: np.ndarray):
    """验证单一姿态的ICP残差（不再尝试镜像翻转）,作用是：给定一个位姿（旋转 + 平移），把模型点云按这个位姿变换到场景坐标系下，然后计算变换后的模型点与场景点之间的距离残差（通常取中位数）"""
    R0 = pose_in[:3, :3].astype(np.float64, copy=True)
    t0 = pose_in[:3, 3].astype(np.float64, copy=True)

    M = np.asarray(model_xyz, dtype=np.float64)           # 模型的点云数据
    S = np.asarray(scene_xyz, dtype=np.float64)           # S表示场景点云

    Mtf = (M @ R0.T) + t0                                 # 模型点云经过姿态变换后的坐标
    med = _median_nn_dist(Mtf, S)                         # 计算变化后的误差

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

    M = model_xyz.astype(np.float32)                                # 转成float32
    S = scene_xyz.astype(np.float32)

    for i, (P, v) in enumerate(zip(poses, votes)):                  # zip 将 poses 与 votes 配对；enumerate 给出下标 i
        init_pose = P.astype(np.float32)
        R = init_pose[:3, :3]
        t = init_pose[:3, 3]
        M_init = (M @ R.T) + t

        out = icp.registerModelToScene(M_init.astype(np.float32), S.astype(np.float32))        #执行icp的精匹配
        icp_pose, _ = _parse_icp_output(out)

        cand1 = icp_pose @ init_pose                               # icp*init  先用PPF粗对齐获取位姿，现在就是icp得到微调，icp_pose @ init_pose代表粗匹配的基础上进行微调
        _, res1 = _try_pose_variants(cand1, M, S)                  # 计算残差
        cand1_fix = cand1
        print(f"  - cand #{i:02d}: votes={v}, icp_res={res1}, use=icp*init")
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


#---------计算质心的距离-------------
def overlap_centroids_kdtree(model_xyz: np.ndarray,
                             scene_xyz: np.ndarray,
                             pose_4x4: np.ndarray,
                             radius: float,
                             min_nn: int = 3,
                             mutual_check: bool = False,
                             return_indices: bool = False,
                             unique_scene: bool = True
                             ) -> Union[
                                 Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                             ]:
    """
    用 KDTree 半径检索找到重叠区域，并计算质心。
    - 模型质心：只对“有邻居”的模型点做普通平均（不按邻居数加权）。
    - 场景质心：默认对命中的场景点“去重后平均”（unique_scene=True）。
      如果 unique_scene=False，则对所有邻居点做重复累计（和之前类似）。
    """

    M = model_xyz.astype(np.float32, copy=False)
    S = scene_xyz.astype(np.float32, copy=False)

    # 将模型点变换到相机/场景坐标
    R = pose_4x4[:3, :3].astype(np.float32)
    t = pose_4x4[:3, 3].astype(np.float32)
    M_tf = (M @ R.T) + t  # (M,3)

    tree_S = cKDTree(S)
    if mutual_check:
        tree_M = cKDTree(M_tf)

    # —— 模型质心（普通平均，不加权）——
    sum_model = np.zeros(3, dtype=np.float64)
    cnt_model = 0

    # —— 场景质心（默认去重平均）——
    if unique_scene:
        scene_hit = set()  # 记录命中的场景点索引（去重）
        # 如果需要把“参与平均的场景点”索引也返回，可用这个 set
    else:
        sum_scene = np.zeros(3, dtype=np.float64)
        cnt_scene = 0

    # 需要返回索引的话，这两个会塞入
    idx_model_kept = [] if return_indices else None
    idx_scene_kept = [] if return_indices else None

    for i, p in enumerate(M_tf):
        neigh = tree_S.query_ball_point(p, r=radius)
        if len(neigh) < min_nn:
            continue

        if mutual_check:
            back = tree_M.query_ball_point(S[neigh], r=radius)
            neigh = [j for j, blist in zip(neigh, back) if i in blist]
            if len(neigh) < min_nn:
                continue

        # —— 模型：该模型点“有效”，计入普通平均 ——
        sum_model += p.astype(np.float64)
        cnt_model += 1
        if return_indices:
            idx_model_kept.append(i)

        # —— 场景：命中的场景点 ——
        if unique_scene:
            # 去重统计
            scene_hit.update(neigh)
        else:
            # 重复累计（旧逻辑）
            ptsS = S[neigh]  # (k,3)
            sum_scene += ptsS.sum(axis=0, dtype=np.float64)
            cnt_scene += len(neigh)

        if return_indices:
            # 注意：这里按“模型点重复记录邻居索引”，主要用于可视化连线；
            # 如果你更想返回“去重后的场景索引”，可以后面把 scene_hit 转成数组返回。
            idx_scene_kept.extend(neigh)

    # —— 结果合成 ——
    if cnt_model == 0:
        raise RuntimeError("没有满足半径与最小邻居数的重叠点，调大 radius 或减小 min_nn 再试。")

    # 模型质心：普通平均
    ctr_model = (sum_model / cnt_model).astype(np.float32)

    # 场景质心：去重平均（默认）或重复累计
    if unique_scene:
        if len(scene_hit) == 0:
            raise RuntimeError("没有命中的场景点（unique_scene=True）。请调参。")
        hit_idx = np.fromiter(scene_hit, dtype=np.int32)
        ctr_scene = S[hit_idx].mean(axis=0).astype(np.float32)
        scene_indices_for_return = hit_idx  # 去重后的场景索引
    else:
        if cnt_scene == 0:
            raise RuntimeError("没有命中的场景点（unique_scene=False）。请调参。")
        ctr_scene = (sum_scene / cnt_scene).astype(np.float32)
        scene_indices_for_return = np.asarray(idx_scene_kept, dtype=np.int32) if return_indices else None

    out = (ctr_scene, ctr_model)
    if return_indices:
        # 模型索引是“有效模型点”的索引（去重），场景索引按 unique_scene 选择去重/不去重
        out += (
            scene_indices_for_return,
            np.asarray(idx_model_kept, dtype=np.int32),
        )
    return out



def _make_sphere(center, radius=0.01, color=(1.0, 0.2, 0.2)):
    """Open3D 画一个小球标记点。"""
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sp.compute_vertex_normals()
    sp.paint_uniform_color(color)
    sp.translate(center.astype(float))
    return sp


def show_with_centroids(scene_pc: o3d.geometry.PointCloud,
                        model_aligned: o3d.geometry.PointCloud,
                        ctr_scene: np.ndarray,
                        ctr_model: np.ndarray,
                        title: str = "Overlap centroids"):
    """可视化：场景+对齐后的模型 + 两个质心小球 + 连接线"""
    g = []
    g.append(scene_pc.paint_uniform_color([0.0, 0.6, 1.0]))
    g.append(model_aligned.paint_uniform_color([1.0, 0.2, 0.2]))

    sp_scene = _make_sphere(ctr_scene, radius=0.015, color=(0.0, 1.0, 0.0))
    sp_model = _make_sphere(ctr_model, radius=0.015, color=(1.0, 0.8, 0.0))
    g.extend([sp_scene, sp_model])

    # 两个质心之间的连线（可选）
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([ctr_scene, ctr_model])),
        lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
    )
    line.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]], dtype=float))
    g.append(line)

    o3d.visualization.draw_geometries(g, window_name=title)


def show_separate_clouds(scene_pc: o3d.geometry.PointCloud,
                         model_pc: o3d.geometry.PointCloud,
                         ctr_scene=None,
                         ctr_model=None):
    """
    分别显示场景点云和模型点云（带可选质心）。
    """
    # 1️⃣ 显示场景
    objs_scene = [scene_pc.paint_uniform_color([0.0, 0.6, 1.0])]
    if ctr_scene is not None:
        objs_scene.append(_make_sphere(ctr_scene, radius=0.015, color=(0.0, 1.0, 0.0)))
    o3d.visualization.draw_geometries(objs_scene, window_name="Scene Cloud (blue + green centroid)")

    # 2️⃣ 显示模型
    objs_model = [model_pc.paint_uniform_color([1.0, 0.2, 0.2])]
    if ctr_model is not None:
        objs_model.append(_make_sphere(ctr_model, radius=0.015, color=(1.0, 0.8, 0.0)))
    o3d.visualization.draw_geometries(objs_model, window_name="Model Cloud (red + yellow centroid)")
