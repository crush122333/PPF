# -*- coding: utf-8 -*-
"""
ppf_icp_main.py
主程序：读取点云 → 预处理 → PPF 训练与匹配 → 候选筛选 → ICP(+镜像修正) → 可视化/保存
依赖：opencv-contrib-python, open3d, numpy
"""

import numpy as np
import open3d as o3d
import cv2 as cv
from ppf_icp_utils import (
    estimate_normals_consistent_knn,
    to_ppf_array_patch,
    run_icp_for_candidates,
    extract_candidates,
    debug_get_props,
    transform_cloud,
    show_two_clouds,
    unify_normals_orientation,
    overlap_centroids_kdtree,
    show_with_centroids,
    show_separate_clouds,

)

# =============== 路径 ===============
MODEL_PATH = "cuboid_model.ply"
SCENE_PATH = "cube_scene_crop.ply"

# =============== 参数（与 MATLAB 对齐） ===============
voxel_model = 0.01
voxel_scene = 0.01

normal_radius_m = 3.0 * voxel_model
normal_radius_s = 3.0 * voxel_scene
_eps = 1e-12

# MATLAB: k = max(10, round((radius/voxel)^2))
kn_m = max(10, int(round((normal_radius_m / max(voxel_model, _eps)) ** 2)))
kn_s = max(10, int(round((normal_radius_s / max(voxel_scene, _eps)) ** 2)))

# PPF detector 参数
PPF_REL_SAMP_STEP = 0.05
PPF_REL_DIST_STEP = 0.05
PPF_NUM_ANGLES    = 45

# ICP 参数
ICP_MAX_ITER  = 200
ICP_TOL       = 1e-5
ICP_REJ_SCALE = 2.0
ICP_LEVELS    = 5

# Patch（PCA法向&曲率）筛选
PATCH_RADIUS_M   = 4.0 * voxel_model
PATCH_RADIUS_S   = 4.0 * voxel_scene
PATCH_MIN_PTS    = 20
CURV_MIN         = 0.001
CURV_MAX         = 0.30
PATCH_STRIDE     = 1
PATCH_VOXEL_MERGE= 0.0

# 候选选择策略
SELECT_MODE  = "votes"   # "votes" | "residual" | "hybrid"
TOPK_VOTES   = 30
TOPK_RESID   = 15


def main(model_path=MODEL_PATH, scene_path=SCENE_PATH, show=True, save_npz="ppf_icp_result.npz"):
    print("[OpenCV] version:", cv.__version__)

    # 读点云
    model_pc = o3d.io.read_point_cloud(model_path)
    scene_pc = o3d.io.read_point_cloud(scene_path)
    print(f"[Read] model: {len(model_pc.points)} pts, scene: {len(scene_pc.points)} pts")

    # 下采样
    model_ds = model_pc.voxel_down_sample(voxel_model)
    scene_ds = scene_pc.voxel_down_sample(voxel_scene)
    print(f"[Voxel] model: {len(model_pc.points)} -> {len(model_ds.points)}, "
          f"scene: {len(scene_pc.points)} -> {len(scene_ds.points)}")

    # 去噪（与 MATLAB 对齐：更宽松）
    model_dn = model_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    scene_dn = scene_ds.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)[0]
    print(f"[Denoise] model: {len(model_dn.points)}, scene: {len(scene_dn.points)}")

    # 法向一致化（k 近邻，与 MATLAB 等价） 后面是场景点云和模型点云的朝向一致化,法向在一个方向
    model_dn = estimate_normals_consistent_knn(model_dn, kn_m)
    scene_dn = estimate_normals_consistent_knn(scene_dn, kn_s)
    model_dn, scene_dn = unify_normals_orientation(model_dn, scene_dn)      #统一朝向

    # 构造 Nx6（Patch PCA+曲率筛选）
    model_ppf = to_ppf_array_patch(model_dn, PATCH_RADIUS_M, PATCH_MIN_PTS,
                                   CURV_MIN, CURV_MAX, PATCH_STRIDE, PATCH_VOXEL_MERGE)
    scene_ppf = to_ppf_array_patch(scene_dn, PATCH_RADIUS_S, PATCH_MIN_PTS,
                                   CURV_MIN, CURV_MAX, PATCH_STRIDE, PATCH_VOXEL_MERGE)
    print(f"[PPF Arrays(PATCH)] model: {model_ppf.shape[0]} x 6, scene: {scene_ppf.shape[0]} x 6")
    if model_ppf.shape[0] == 0 or scene_ppf.shape[0] == 0:
        raise RuntimeError("PPF arrays 为空（patch 过滤后无点）。请调小 curvature_min/增大 patch_radius。")

    # PPF 训练+匹配
    det = cv.ppf_match_3d.PPF3DDetector(PPF_REL_SAMP_STEP, PPF_REL_DIST_STEP, int(PPF_NUM_ANGLES))     # 先设置PPF的参数
    det.trainModel(model_ppf.astype(np.float32))                                                       # 先对模型的点数进行训练
    py_res = det.match(scene_ppf.astype(np.float32))                                                   # 在对场景的点云进行一个匹配

    debug_get_props(py_res)                                                                            # 对训练后得到的返回值进行解析
    nC, poses, votes, ppf_residuals = extract_candidates(py_res)                                       # 提取训练的结果，投票数，残差以及位姿
    print(f"[PPF] candidates: {nC}")
    if nC == 0:
        raise RuntimeError("No PPF matches found.")

    # 候选筛选
    poses_f   = poses                                                                         # 位姿矩阵
    votes_f   = votes                                                                         # PPF投票数
    ppf_res_f = ppf_residuals                                                                 # PPF粗匹配的残差

    # 按投票数降序排列（np.argsort 默认升序，加负号即可降序）
    order = np.argsort(-votes_f)

    # 取前 K 个（TOPK_VOTES 是保留的候选数量，比如30）
    K = min(TOPK_VOTES, len(order))
    sel = order[:K]                                                                           # sel作为索引号进行筛选 排序

    # 选出对应的位姿和投票数
    poses_sel = [poses_f[i] for i in sel]  # poses 是 list
    votes_sel = votes_f[sel]  # votes 是 ndarray

    print(f"[SELECT] by votes → {len(poses_sel)} candidates into ICP")

    # ICP 循环 + 自动姿态修正
    best_pose, best_res, best_idx, logs = run_icp_for_candidates(
        model_ppf[:, :3], scene_ppf[:, :3], poses_sel, votes_sel,
        icp_max_iter=ICP_MAX_ITER, icp_tol=ICP_TOL,
        icp_rej_scale=ICP_REJ_SCALE, icp_levels=ICP_LEVELS
    )

    print(f"\n[FINAL POSE] residual = {best_res} (from candidate #{best_idx})\n{best_pose}")

    # ===== 计算重叠区域质心，直接对重叠的模型点云进行质心的计算=====
    try:
        ctr_scene, ctr_model = overlap_centroids_kdtree(
            model_xyz=model_ppf[:, :3],
            scene_xyz=scene_ppf[:, :3],
            pose_4x4=best_pose,
            radius=3.0 * voxel_scene,
            min_nn=3,
            mutual_check=False,
            unique_scene=True,  # 场景点去重后平均（推荐）
            # return_indices=True,      # 如需拿到索引就加上
        )

        dist_model = float(np.linalg.norm(ctr_model))  # 相机→模型质心（理论）
        dist_scene = float(np.linalg.norm(ctr_scene))  # 相机→场景质心（观测）
        print(f"[CENTROID] cam->model-centroid = {dist_model:.4f} m, "
              f"cam->scene-centroid = {dist_scene:.4f} m")
    except Exception as e:
        ctr_scene = ctr_model = None
        print("[CENTROID] 计算失败：", e)

    # 保存
    np.savez(save_npz, final_pose=best_pose, best_residual=best_res, best_index=best_idx, ctr_scene=ctr_scene, ctr_model=ctr_model)

    # 可视化（仅最优）
    if show:
        model_aligned = transform_cloud(model_dn, best_pose)
        show_two_clouds(scene_dn, model_aligned, title=f"PPF+ICP 最优 #{best_idx} (res={best_res:.4g})")

    if show and (ctr_scene is not None) and (ctr_model is not None):
        show_with_centroids(scene_dn, model_aligned,
                            ctr_scene=ctr_scene, ctr_model=ctr_model,
                            title="Overlap centroids (green=scene, yellow=model)")

    # # ===== 单独显示模型和场景（不叠加） =====
    # show_separate_clouds(scene_dn, model_aligned,
    #                      ctr_scene=ctr_scene, ctr_model=ctr_model)


if __name__ == "__main__":
    main()
