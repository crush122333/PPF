#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, platform, time, copy, re
import numpy as np

# ---------- 路径与依赖 ----------
BASE = os.path.abspath(os.path.dirname(__file__))
def _add_sdk_path():
    if platform.system() == "Windows":
        p = os.path.join(BASE, ".", "lib", "win32")
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
        sys.path.append(p)
    # else:
    #     arch = platform.machine()
    #     if arch == "x86_64":
    #         sys.path.append(os.path.join(BASE, "..", "lib", "linux", "x86_64"))
    #     elif arch == "aarch64":
    #         sys.path.append(os.path.join(BASE, "..", "lib", "linux", "aarch64"))

_add_sdk_path()
sys.path.append(os.path.join(BASE, ".", "cfg"))

import xintan_sdk
from read_config import ConfigParse


# ---------- 小工具 ----------
def _extract_version(s: str):
    m = re.search(r'[vV](\d+\.\d+)(?:\.\d+)?', s or "")
    if not m: return None
    try: return float(m.group(1))
    except: return None

def _parse_enum(enum_cls, val):
    """cfg里的 'ModulationFreq.FREQ_24M' / 'FREQ_24M' / 数字 -> 枚举"""
    if val is None: return None
    if isinstance(val, enum_cls): return val
    if isinstance(val, str):
        name = val.split('.')[-1].strip()
        return getattr(enum_cls, name, None)
    try:
        return enum_cls(int(val))
    except Exception:
        return None

def _apply_sdk_filters(sdk: xintan_sdk.XtSdk, filters: dict | None):
    f = filters or {}
    def g(k, d=None): return f.get(k, d)

    # 中值
    if int(g("medianEnable", g("medianSize", 0) > 0)) == 1 and int(g("medianSize", 0)) in (3, 5):
        if hasattr(sdk, "setSdkMedianFilter"):
            sdk.setSdkMedianFilter(int(g("medianSize")))
    # 卡尔曼
    if int(g("kalmanEnable", 0)) == 1 and hasattr(sdk, "setSdkKalmanFilter"):
        sdk.setSdkKalmanFilter(int(float(g("kalmanFactor", 0.3)) * 1000),
                               int(g("kalmanThreshold", 300)),
                               2000)
    # 边沿
    if int(g("edgeEnable", 0)) == 1 and hasattr(sdk, "setSdkEdgeFilter"):
        sdk.setSdkEdgeFilter(int(g("edgeThreshold", 60)))
    # 尘埃
    if int(g("dustEnable", 0)) == 1 and hasattr(sdk, "setSdkDustFilter"):
        sdk.setSdkDustFilter(int(g("dustThreshold", 9000)),
                             int(g("dustFrames", 2)))
    # 后处理（动态）
    if int(g("postprocessEnable", 0)) == 1 and hasattr(sdk, "setPostProcess"):
        sdk.setPostProcess(int(g("postprocessThreshold", 5)),
                           int(g("dynamicsEnabled", 0)),
                           int(g("dynamicsWinsize", 9)))
    # 反射性抑制
    if int(g("reflectiveEnable", 0)) == 1 and hasattr(sdk, "setSdkReflectiveFilter"):
        sdk.setSdkReflectiveFilter(float(g("ref_th_min", 0.5)),
                                   float(g("ref_th_max", 2.0)))


# ---------- 对外主函数 ----------
def capture_scene_pointcloud(
    ip: str = "192.168.1.113",
    out_dir: str = "./xt_pointcloud_out",
    max_dist: float | None = 2.0,
    img_type: str | None = None,   # 可传 "IMG_POINTCLOUD" / "IMG_POINTCLOUDAMP"；None 用 cfg
    timeout_s: float = 8.0,
) -> np.ndarray:
    """
    连接相机并采集一帧点云，使用 SDK 内置滤波，保存为 out_dir/scene.pcd，
    返回 Nx4 numpy 数组 [x,y,z,intensity]（不画图）。
    """
    # 读取配置
    cfg = ConfigParse(os.path.join(BASE, "..", "cfg", "xintan.xtcfg")).configs
    cfg = copy.deepcopy(cfg)
    S = cfg.get("Setting", {})
    F = cfg.get("Filters", {})

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(os.path.abspath(out_dir), "scene.pcd")

    sdk = xintan_sdk.XtSdk()

    # 回调拿数据
    buf = {"xyzi": None}
    start_ts = {"t": None}

    def _on_state(ev: xintan_sdk.CBEventData):
        if ev.eventstr != "sdkState": return
        try:
            if sdk.isconnect() and ev.cmdid == 0xfe:
                ok, devinfo = sdk.getDevInfo()
                fwf = _extract_version(devinfo.fwVersion) if ok else None

                # （可选）把设备侧当前配置合并到本地显示
                if fwf and fwf >= 2.20:
                    ok_cfg, devcfg = sdk.getDevConfig()
                    if ok_cfg:
                        S['int1'], S['int2'], S['int3'], S['int4'] = devcfg.integrationTimes
                        S['HDR']  = devcfg.hdrMode
                        S['freq'] = devcfg.modFreq

                # 设置参数 + 滤波 + start
                sdk.stop()
                sdk.setIntTimesus(int(S.get('intgs', 1)),
                                  int(S.get('int1', 1000)),
                                  int(S.get('int2', 0)),
                                  int(S.get('int3', 0)),
                                  int(S.get('int4', 0)), 0)
                sdk.setHdrMode(xintan_sdk.HDRMode(int(S.get('HDR', 0))))
                sdk.setMinAmplitude(int(S.get('minLSB', 120)))
                sdk.setMaxFps(int(S.get('maxfps', 15)))

                # 调制频率（cfg 可为 "ModulationFreq.FREQ_24M" 或 "FREQ_24M"）
                if hasattr(sdk, 'setModFreq') and 'freq' in S:
                    mf = _parse_enum(xintan_sdk.ModulationFreq, S['freq'])
                    if mf is not None:
                        sdk.setModFreq(mf)

                _apply_sdk_filters(sdk, F)

                it_name = (img_type or S.get('imgType') or 'IMG_POINTCLOUD').upper()
                it = getattr(xintan_sdk.ImageType, it_name, xintan_sdk.ImageType.IMG_POINTCLOUD)
                sdk.start(it)
        except Exception as e:
            print("[STATE] err:", e)

    def _on_image(frame: xintan_sdk.Frame):
        # 只取第一帧
        if buf["xyzi"] is not None:
            return
        try:
            n = len(frame.points)
            if n == 0:
                return
            x = np.fromiter((p.x for p in frame.points), dtype=np.float32, count=n)
            y = np.fromiter((p.y for p in frame.points), dtype=np.float32, count=n)
            z = np.fromiter((p.z for p in frame.points), dtype=np.float32, count=n)
            amp = np.array(frame.amplData, dtype=np.float32) if hasattr(frame, "amplData") and len(frame.amplData) == n \
                  else np.zeros(n, np.float32)

            if max_dist is not None:
                dist = np.sqrt(x*x + y*y + z*z)
                mask = dist <= float(max_dist)
                x, y, z, amp = x[mask], y[mask], z[mask], amp[mask]

            buf["xyzi"] = np.column_stack([x, y, z, amp])
        except Exception as e:
            print("[IMAGE] err:", e)

    # 连接并启动
    sdk.setCallback(_on_state, _on_image)
    sdk.setConnectIpaddress(ip)
    sdk.startup()

    # 简单等待直到拿到一帧或超时
    start_ts["t"] = time.time()
    while buf["xyzi"] is None and (time.time() - start_ts["t"]) < timeout_s:
        time.sleep(0.01)

    # 收尾
    try:
        sdk.stop()
        time.sleep(0.1)
        sdk.shutdown()
    except Exception:
        pass

    xyzi = buf["xyzi"]
    if xyzi is None:
        raise TimeoutError("采集超时：未获得点云帧，请检查网络/供电/防火墙。")

    # 保存 scene.pcd（ASCII）
    with open(save_path, "w") as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {xyzi.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {xyzi.shape[0]}\nDATA ascii\n")
        np.savetxt(f, xyzi, fmt="%.6f %.6f %.6f %.6f")

    # 控制台简要信息（不画图）
    print(f"[OK] points={xyzi.shape[0]}  saved: {save_path}")

    return xyzi
