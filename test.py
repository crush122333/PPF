from xt_pc_single import capture_scene_pointcloud
import numpy as np
import open3d as o3d

def remove_plane_and_denoise(
    pts: np.ndarray,
    *,
    plane_dist: float = 0.008,   # RANSAC 距离阈值(米)
    ransac_n: int = 3,
    num_iter: int = 2000,
    keep: str = "nonplane",      # "nonplane" 保留非平面(常用)；"plane" 仅保留平面
    use_stat: bool = True,       # 统计滤波
    stat_nb: int = 20,
    stat_std: float = 1.5,
    use_radius: bool = False,    # 半径滤波 (更严格，点稀疏时慎用)
    radius: float = 0.02,
    min_pts: int = 16,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    pts: (N,3) 或 (N,4)，列为 [x, y, z, (intensity)]
    return: (filtered_pts, plane_model) 其中 plane_model=(a,b,c,d) 表示 ax+by+cz+d=0
    """
    assert pts.ndim == 2 and pts.shape[1] in (3, 4), "pts 需为 (N,3) 或 (N,4)"
    xyz = pts[:, :3].astype(np.float64)

    # 1) RANSAC 找最大平面
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd_all.segment_plane(distance_threshold=plane_dist,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iter)
    inliers = np.asarray(inliers, dtype=np.int64)

    # 2) 选出保留的索引
    all_idx = np.arange(xyz.shape[0], dtype=np.int64)
    if keep == "plane":
        keep_idx = inliers
    else:  # 默认：保留非平面点
        mask = np.ones_like(all_idx, dtype=bool)
        mask[inliers] = False
        keep_idx = all_idx[mask]

    # 3) （可选）对保留集合做降噪滤波
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[keep_idx])

    if use_stat:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=stat_nb, std_ratio=stat_std)
        keep_idx = keep_idx[np.asarray(ind, dtype=np.int64)]

    if use_radius:
        pcd, ind = pcd.remove_radius_outlier(nb_points=min_pts, radius=radius)
        keep_idx = keep_idx[np.asarray(ind, dtype=np.int64)]

    # 4) 返回同形状列（保留强度）
    filtered = pts[keep_idx]
    return filtered.astype(np.float32, copy=False), tuple(map(float, plane_model))


def show_pointcloud_o3d(pts: np.ndarray, *, color_by: str = "z", voxel: float | None = None, title: str = "scene"):
    """
    最简可视化：color_by = "z" / "intensity" / "uniform"
    """
    xyz = pts[:, :3].astype(np.float64)
    inten = pts[:, 3].astype(np.float64) if pts.shape[1] == 4 else None

    if color_by == "intensity" and inten is not None:
        v = inten
        v = (v - v.min()) / (np.ptp(v) + 1e-9)  # NumPy 2.0 用 np.ptp(v)
        colors = np.stack([v, 1 - np.abs(v - 0.5)*2, 1 - v], axis=1)
    elif color_by == "z":
        v = xyz[:, 2]
        v = (v - v.min()) / (np.ptp(v) + 1e-9)
        colors = np.stack([0.1 + 0.9*v, 0.9*(1 - v), 0.2 + 0.6*v], axis=1)
    else:
        colors = np.tile(np.array([[0.9, 0.1, 0.1]]), (xyz.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    o3d.visualization.draw_geometries([pcd], window_name=title)




# 1) 采一帧（返回 Nx4：x,y,z,intensity），并保存到 ./xt_pointcloud_out/scene.pcd
pts = capture_scene_pointcloud(
    ip="192.168.1.113",
    out_dir="./xt_pointcloud_out",
    max_dist=2.0,
    img_type=None,   # 用 cfg；也可写 "IMG_POINTCLOUD" / "IMG_POINTCLOUDAMP"
    timeout_s=8.0,
)

print("shape:", pts.shape)
print("first 5 rows:\n", pts[:5])

# 先看原始
show_pointcloud_o3d(pts, color_by="z", voxel=None, title="Raw")

# 去平面 + 降噪（常用配置）
clean, plane = remove_plane_and_denoise(
    pts,
    plane_dist=0.008,   # 你场景单位是米，地面/桌面平整就 5~10mm
    ransac_n=3,
    num_iter=2000,
    keep="nonplane",    # 保留非平面
    use_stat=True, stat_nb=20, stat_std=1.5,
    use_radius=False,   # 点非常稠密再开，或配合小半径
)
print("plane model:", plane)
print("clean shape:", clean.shape)

# 看过滤结果
show_pointcloud_o3d(clean, color_by="z", voxel=0.005, title="Non-plane + Denoised")


