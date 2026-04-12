import csv
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from importlib.util import find_spec


INPUT_DIR = Path("dataset/drink_water")
OUTPUT_ROOT = INPUT_DIR.parent / f"{INPUT_DIR.name}_split"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ENABLE_IMU = False
FPS_FALLBACK = 30
PROCESS_BAG = True
PROCESS_EVENT = False
SKIP_EXISTING = True


def safe_get_fps(profile, fallback=30):
    try:
        return int(profile.fps())
    except Exception:
        return fallback


def depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)


def ir_to_bgr(ir_image: np.ndarray) -> np.ndarray:
    if ir_image.dtype != np.uint8:
        ir_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
        ir_u8 = ir_norm.astype(np.uint8)
    else:
        ir_u8 = ir_image
    return cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR)


def realsense_color_to_bgr(color_frame: rs.video_frame) -> np.ndarray:
    color_image = np.asanyarray(color_frame.get_data())
    color_format = color_frame.profile.format()

    if color_format == rs.format.rgb8:
        return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    if color_format == rs.format.bgr8:
        return color_image

    if color_format == rs.format.rgba8:
        return cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)

    if color_format == rs.format.bgra8:
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    if color_format == rs.format.yuyv:
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)

    if color_format == rs.format.y16:
        return ir_to_bgr(color_image)

    if color_image.ndim == 2:
        return cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)

    if color_image.ndim == 3 and color_image.shape[2] == 4:
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    if color_image.ndim == 3 and color_image.shape[2] == 3:
        # Unknown three-channel format. Keep the raw channel order rather than
        # swapping channels blindly and risking another red/blue inversion.
        return color_image

    raise ValueError(f"Unsupported RealSense color format: {color_format}")


def write_imu_csv(rows, csv_path: Path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "sensor", "x", "y", "z"])
        writer.writerows(rows)


def create_imu_video_from_csv(csv_path: Path, out_mp4: Path, fps=20):
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "timestamp_ms": float(row["timestamp_ms"]),
                    "sensor": row["sensor"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                }
            )

    if not data:
        print("No IMU data found, skip imu.mp4 generation.")
        return

    accel = [d for d in data if d["sensor"] == "accel"]
    gyro = [d for d in data if d["sensor"] == "gyro"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)

    tmp_dir = out_mp4.parent / "_imu_frames"
    tmp_dir.mkdir(exist_ok=True)

    frame_paths = []
    n_steps = max(len(accel), len(gyro))

    for i in range(1, n_steps + 1):
        axes[0].clear()
        axes[1].clear()

        cur_accel = accel[: min(i, len(accel))]
        cur_gyro = gyro[: min(i, len(gyro))]

        if cur_accel:
            t = [x["timestamp_ms"] for x in cur_accel]
            axes[0].plot(t, [x["x"] for x in cur_accel], label="ax")
            axes[0].plot(t, [x["y"] for x in cur_accel], label="ay")
            axes[0].plot(t, [x["z"] for x in cur_accel], label="az")
            axes[0].set_title("Accelerometer")
            axes[0].legend()
            axes[0].set_xlabel("timestamp_ms")

        if cur_gyro:
            t = [x["timestamp_ms"] for x in cur_gyro]
            axes[1].plot(t, [x["x"] for x in cur_gyro], label="gx")
            axes[1].plot(t, [x["y"] for x in cur_gyro], label="gy")
            axes[1].plot(t, [x["z"] for x in cur_gyro], label="gz")
            axes[1].set_title("Gyroscope")
            axes[1].legend()
            axes[1].set_xlabel("timestamp_ms")

        frame_path = tmp_dir / f"{i:06d}.png"
        plt.savefig(frame_path)
        frame_paths.append(frame_path)

    plt.close(fig)

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for p in frame_paths:
        img = cv2.imread(str(p))
        writer.write(img)

    writer.release()

    for p in frame_paths:
        p.unlink()
    tmp_dir.rmdir()


def sample_name_from_bag(bag_path: Path) -> str:
    return bag_path.stem.replace("_realsense", "")


def prepare_dvsense_import():
    spec = find_spec("dvsense_driver")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("dvsense_driver is not installed")

    package_dir = Path(list(spec.submodule_search_locations)[0])
    search_dirs = [package_dir, package_dir / "base", package_dir / "hal"]

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for path in search_dirs:
            os.add_dll_directory(str(path))
    else:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([*(str(p) for p in search_dirs), current_path])


def export_raw_to_video(sample_name: str):
    raw_path = INPUT_DIR / f"{sample_name}_event.raw"
    event_mp4 = OUTPUT_ROOT / f"{sample_name}_event.mp4"

    if not raw_path.exists():
        print(f"[WARN] Missing raw file for {sample_name}: {raw_path.name}")
        return

    if SKIP_EXISTING and event_mp4.exists():
        print(f"[SKIP] Event video already exists for {sample_name}: {event_mp4.name}")
        return

    try:
        prepare_dvsense_import()
        from dvsense_driver.raw_file_reader import RawFileReader
    except Exception as exc:
        print(f"[WARN] Failed to import dvsense_driver for {raw_path.name}: {exc}")
        return

    reader = None
    try:
        reader = RawFileReader(str(raw_path))
        if not reader.load_file():
            print(f"[WARN] Failed to load raw file: {raw_path.name}")
            return

        start_ok, start_ts = reader.get_start_timestamp()
        end_ok, end_ts = reader.get_end_timestamp()
        if not start_ok or not end_ok:
            print(f"[WARN] Failed to get timestamp range for {raw_path.name}")
            return

        exported = reader.export_event_to_video(int(start_ts), int(end_ts), str(event_mp4))
        if not exported:
            print(f"[WARN] SDK failed to export video for {raw_path.name}")
            return

        if event_mp4.exists():
            print(f"[OK] Exported {raw_path.name} -> {event_mp4}")
        else:
            print(
                f"[WARN] SDK reported success for {raw_path.name}, "
                f"but expected output was not found: {event_mp4.name}"
            )
    except Exception as exc:
        print(f"[WARN] Failed to export {raw_path.name} to video: {exc}")
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def convert_bag_file(bag_path: Path, sample_name: str):
    rgb_mp4 = OUTPUT_ROOT / f"{sample_name}_rgb.mp4"
    depth_mp4 = OUTPUT_ROOT / f"{sample_name}_depth.mp4"
    ir_mp4 = OUTPUT_ROOT / f"{sample_name}_ir.mp4"
    imu_csv = OUTPUT_ROOT / f"{sample_name}_imu.csv"
    imu_mp4 = OUTPUT_ROOT / f"{sample_name}_imu.mp4"

    if SKIP_EXISTING and rgb_mp4.exists() and depth_mp4.exists() and ir_mp4.exists():
        print(f"[SKIP] Bag outputs already exist for {sample_name}")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)

    try:
        config.enable_stream(rs.stream.color)
    except Exception:
        pass
    try:
        config.enable_stream(rs.stream.depth)
    except Exception:
        pass
    try:
        config.enable_stream(rs.stream.infrared, 1)
    except Exception:
        pass
    if ENABLE_IMU:
        try:
            config.enable_stream(rs.stream.accel)
        except Exception:
            pass
        try:
            config.enable_stream(rs.stream.gyro)
        except Exception:
            pass

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    rgb_writer = None
    depth_writer = None
    ir_writer = None
    imu_rows = []

    color_size = None
    depth_size = None
    ir_size = None

    rgb_fps = FPS_FALLBACK
    depth_fps = FPS_FALLBACK
    ir_fps = FPS_FALLBACK
    color_format = None

    try:
        active_profile = pipeline.get_active_profile()

        for sp in active_profile.get_streams():
            if sp.stream_type() == rs.stream.color:
                vsp = sp.as_video_stream_profile()
                color_size = (vsp.width(), vsp.height())
                rgb_fps = safe_get_fps(vsp, FPS_FALLBACK)
                color_format = sp.format()
            elif sp.stream_type() == rs.stream.depth:
                vsp = sp.as_video_stream_profile()
                depth_size = (vsp.width(), vsp.height())
                depth_fps = safe_get_fps(vsp, FPS_FALLBACK)
            elif sp.stream_type() == rs.stream.infrared:
                vsp = sp.as_video_stream_profile()
                ir_size = (vsp.width(), vsp.height())
                ir_fps = safe_get_fps(vsp, FPS_FALLBACK)

        if color_size:
            print(f"[INFO] {bag_path.name}: detected color stream format = {color_format}")
            rgb_writer = cv2.VideoWriter(
                str(rgb_mp4), cv2.VideoWriter_fourcc(*"mp4v"), rgb_fps, color_size
            )

        if depth_size:
            depth_writer = cv2.VideoWriter(
                str(depth_mp4), cv2.VideoWriter_fourcc(*"mp4v"), depth_fps, depth_size
            )

        if ir_size:
            ir_writer = cv2.VideoWriter(
                str(ir_mp4), cv2.VideoWriter_fourcc(*"mp4v"), ir_fps, ir_size
            )

        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            ir_frame = None
            try:
                ir_frame = frames.get_infrared_frame(1)
            except Exception:
                pass

            if color_frame and rgb_writer is not None:
                color_bgr = realsense_color_to_bgr(color_frame)
                rgb_writer.write(color_bgr)

            if depth_frame and depth_writer is not None:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_writer.write(depth_to_colormap(depth_image))

            if ir_frame and ir_writer is not None:
                ir_image = np.asanyarray(ir_frame.get_data())
                ir_writer.write(ir_to_bgr(ir_image))

            if ENABLE_IMU:
                for i in range(frames.size()):
                    frame = frames[i]
                    stream_type = frame.profile.stream_type()
                    ts = frame.get_timestamp()

                    if stream_type == rs.stream.accel:
                        motion = frame.as_motion_frame().get_motion_data()
                        imu_rows.append([ts, "accel", motion.x, motion.y, motion.z])
                    elif stream_type == rs.stream.gyro:
                        motion = frame.as_motion_frame().get_motion_data()
                        imu_rows.append([ts, "gyro", motion.x, motion.y, motion.z])
    finally:
        pipeline.stop()
        if rgb_writer is not None:
            rgb_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        if ir_writer is not None:
            ir_writer.release()

    if ENABLE_IMU:
        write_imu_csv(imu_rows, imu_csv)
        if imu_rows:
            create_imu_video_from_csv(imu_csv, imu_mp4)

    print(f"[OK] Converted {bag_path.name} -> {OUTPUT_ROOT}")


def main():
    bag_files = sorted(INPUT_DIR.glob("*_realsense.bag"))
    if not bag_files:
        raise FileNotFoundError(f"No .bag files found in {INPUT_DIR}")

    for bag_path in bag_files:
        sample_name = sample_name_from_bag(bag_path)
        if PROCESS_BAG:
            convert_bag_file(bag_path, sample_name)
        if PROCESS_EVENT:
            export_raw_to_video(sample_name)


if __name__ == "__main__":
    main()
