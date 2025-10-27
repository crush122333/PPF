# acquisition_xtsdk.py
import os
import sys
import time
import platform
from queue import Queue, Empty
import numpy as np

SDK_DIR = r"D:\project\Python\xintan_cam\xtsdk_py-main\lib\win32"
if platform.system() == "Windows" and os.path.isdir(SDK_DIR):           # 为了导入SDK
    sys.path.append(SDK_DIR)                                            # 把目录加入模块搜索路径，这样 import xintan_sdk 时 Python 才能找到
    os.add_dll_directory(SDK_DIR)

import xintan_sdk
print("SDK 加载成功：", xintan_sdk.__file__)



def _frame_to_xyz(frame) -> np.ndarray:
    """
    兼容多种 SDK 字段：优先使用 hasPointcloud/points，其次 pointcloud/xyz。
    返回 (N,3) float32，单位米。   把SDK的帧结构统一转成（N,3）的点云数据
    """
    # 1) 常见：frame.hasPointcloud + frame.points（C++结构，带 x/y/z）
    if hasattr(frame, "hasPointcloud") and frame.hasPointcloud and hasattr(frame, "points"):
        # points 是一个对象数组：每个元素有 .x/.y/.z
        xyz = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        return xyz

    # 2) 部分版本：frame.pointcloud（扁平 xyz…）
    if hasattr(frame, "pointcloud"):
        arr = np.asarray(frame.pointcloud, dtype=np.float32).reshape(-1, 3)
        return arr

    # 3) 也有 frame.xyz / xyz_float
    for attr in ("xyz", "xyz_float"):
        if hasattr(frame, attr):
            arr = np.asarray(getattr(frame, attr), dtype=np.float32).reshape(-1, 3)
            return arr

    raise RuntimeError("未在 Frame 中找到点云字段（尝试了 hasPointcloud/points、pointcloud、xyz、xyz_float）")


def grab_scene_pointcloud_once_network(ip: str, timeout_s: float = 6.0) -> np.ndarray:      # 返回值是一个数组
    """
    使用官方 xintan_sdk 从网络相机采集**一帧**点云。
    返回：xyz (N,3) float32，单位米。
    """
    xt = xintan_sdk.XtSdk()
    q = Queue(maxsize=2)                                                                  # 线程安全队列，表示采集的帧的图像存放

#--------定义两个回调函数-----------
    def on_image(frame: xintan_sdk.Frame):                                                # 传入的是帧数据
        try:
            if q.full():
                q.get_nowait()                                                            # 判断队列是否满了，满了就丢弃一帧数据，没有满就存放在队列里面
            q.put_nowait(frame)
        except Exception:                                                                 # 异常处理，表示展示异常状态
            pass

    def on_event(evt: xintan_sdk.CBEventData):                                            # 接收 SDK 事件（连接状态、错误码等）。这里先不处理，保留接口
        # 可按需打印 evt.eventstr / evt.cmdid 监控状态
        pass

#------回调函数注册，自己调用-------
    xt.setCallback(on_event, on_image)                                                    # 注册回调：告诉 SDK：事件到了叫 on_event，图像/点云到了叫 on_image。

    if not xt.setConnectIpaddress(ip):                                                    # 尝试设置/连接到相机 IP。若返回 False，抛 RuntimeError。
        raise RuntimeError(f"连接相机失败：{ip}")

    xt.startup()                                                                          # 让 SDK 进入“就绪/工作”状态
    # 选取包含点云的数据流类型（不同固件可能不同，IMG_POINTCLOUD/IMG_POINTCLOUDAMP/或 ImageType(0)）
    try:
        img_type = getattr(xintan_sdk.ImageType, "IMG_POINTCLOUDAMP", xintan_sdk.ImageType(0))
        xt.start(img_type)                #开始接受数据
    except Exception:
        xt.shutdown()
        raise

    # 等一帧
    frame = None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            frame = q.get(timeout=0.05)
            break
        except Empty:
            continue

    # 收尾
    try:
        xt.stop()
    finally:
        xt.shutdown()

    if frame is None:
        raise TimeoutError("等待相机帧超时，请检查网络/IP/供电。")

    xyz = _frame_to_xyz(frame)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def grab_scene_pointcloud_median_network(ip: str, frames: int = 3, timeout_s: float = 6.0) -> np.ndarray:
    """
    采多帧做中值稳噪（要求相机输出为固定尺寸的稠密网格，点数一致）。
    点数不一致时自动回退返回第一帧。
    """
    clouds = [grab_scene_pointcloud_once_network(ip, timeout_s) for _ in range(frames)]
    try:
        X = np.stack(clouds, axis=2)  # (N,3,K)
        return np.nanmedian(X, axis=2).astype(np.float32)
    except Exception:
        return clouds[0]


if __name__ == "__main__":
    # 小测试：仅采一帧并打印点数
    ip = "10.1.1.104"
    pts = grab_scene_pointcloud_once_network(ip)
    print("grabbed points:", pts.shape)
