import os
import numpy as np
import cv2
import pyrealsense2 as rs
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera



# =====================================================================
# MAIN CALIBRATION FUNCTION (called by run())
# =====================================================================
def run_calibration(prefix, dataset_path):
    """
    Calibrate time offset between event camera and RealSense camera.
    Creates <prefix>_delta_seconds.txt inside dataset_path.
    """

    print("\n===============================================")
    print(f" FRAME-LEVEL AUTO CALIBRATION FOR {prefix}")
    print("===============================================\n")

    # Filenames
    event_file = os.path.join(dataset_path, f"{prefix}_event.raw")
    bag_file   = os.path.join(dataset_path, f"{prefix}_realsense.bag")

    # Validate files
    if not os.path.exists(event_file):
        print(f"❌ Event file not found: {event_file}")
        return False

    if not os.path.exists(bag_file):
        print(f"❌ RealSense bag not found: {bag_file}")
        return False

    # -----------------------------------------------------
    # 1. READ EVENT CAM TIME RANGE (RELATIVE)
    # -----------------------------------------------------
    print("Reading event timestamp range...")

    mv = EventsIterator(event_file, delta_t=1000000)
    if not is_live_camera(event_file):
        mv = LiveReplayEventsIterator(mv)

    ev_start = None
    ev_end = None

    for evs in mv:
        if evs is None or len(evs) == 0:
            continue
        if ev_start is None:
            ev_start = evs["t"][0]
        ev_end = evs["t"][-1]

    ev_start_s = ev_start * 1e-6
    ev_end_s   = ev_end * 1e-6

    print(f"Event time range: {ev_start_s:.3f} → {ev_end_s:.3f} sec")

    # -----------------------------------------------------
    # 2. READ REALSENSE TIME RANGE (MAKE RELATIVE!)
    # -----------------------------------------------------
    print("Reading RealSense timestamp range...")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    rs_start_raw = None
    rs_end_rel = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                break

            ts_raw = c.get_timestamp() * 0.001  # ms → sec

            if rs_start_raw is None:
                rs_start_raw = ts_raw

            rel_ts = ts_raw - rs_start_raw   # <<<<< FIXED: make relative time

            rs_end_rel = rel_ts

    except Exception:
        pass

    pipeline.stop()

    print(f"RGB time range (relative): {0.000:.3f} → {rs_end_rel:.3f} sec")

    # -----------------------------------------------------
    # 3. COMPUTE OVERLAP
    # -----------------------------------------------------
    overlap_start = max(ev_start_s, 0.0)
    overlap_end   = min(ev_end_s, rs_end_rel)

    if overlap_end <= overlap_start:
        print("❌ No overlapping time interval. Cannot calibrate.")
        return False

    print(f"Overlap interval: {overlap_start:.3f} → {overlap_end:.3f} sec")

    # -----------------------------------------------------
    # 4. BUILD SIGNAL ARRAYS
    # -----------------------------------------------------
    fs = 1000  # sampling rate (Hz)
    L = int((overlap_end - overlap_start) * fs)

    ev_signal  = np.zeros(L)
    rgb_signal = np.zeros(L)

    # -----------------------------------------------------
    # 5. BUILD EVENT SIGNAL
    # -----------------------------------------------------
    print("Building 1-D event signal...")

    mv = EventsIterator(event_file, delta_t=1000000)
    if not is_live_camera(event_file):
        mv = LiveReplayEventsIterator(mv)

    for evs in mv:
        if evs is None or len(evs) == 0:
            continue

        t = evs["t"] * 1e-6   # convert µs → sec
        mask = (t >= overlap_start) & (t < overlap_end)
        if not np.any(mask):
            continue

        idx = ((t[mask] - overlap_start) * fs).astype(int)
        idx = idx[idx < L]
        np.add.at(ev_signal, idx, 1)

    # -----------------------------------------------------
    # 6. BUILD RGB MOTION SIGNAL
    # -----------------------------------------------------
    print("Building 1-D RGB motion signal...")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    prev_gray = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                break

            ts_raw = c.get_timestamp() * 0.001
            rel_ts = ts_raw - rs_start_raw   # <<<<< FIXED

            if not (overlap_start <= rel_ts < overlap_end):
                continue

            img = np.asanyarray(c.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                idx = int((rel_ts - overlap_start) * fs)
                if 0 <= idx < L:
                    rgb_signal[idx] = np.sum(
                        np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
                    )

            prev_gray = gray

    except Exception:
        pass

    pipeline.stop()

    # -----------------------------------------------------
    # 7. CROSS-CORRELATION
    # -----------------------------------------------------
    print("Computing cross-correlation...")

    a = ev_signal - ev_signal.mean()
    b = rgb_signal - rgb_signal.mean()

    corr = np.correlate(a, b, mode="full")
    lag = corr.argmax() - (L - 1)

    delta_seconds = lag / fs

    print(f"\nEstimated Δt = {delta_seconds:.6f} seconds")
    print("Positive Δt means EVENT stream is ahead of RGB.")

    # -----------------------------------------------------
    # 8. SAVE FILE
    # -----------------------------------------------------
    out_file = os.path.join(dataset_path, f"{prefix}_delta_seconds.txt")

    with open(out_file, "w") as f:
        f.write(f"{delta_seconds:.9f}\n")

    print(f"Saved calibration to: {out_file}")
    print("===============================================\n")

    return True



# =====================================================================
# WRAPPER FUNCTION USED BY MENU
# =====================================================================
def run():
    """
    Wrapper for your main menu system.
    It automatically calibrates prefix '0001' in folder 'dataset'.
    """
    dataset_path = "dataset"
    prefix = "0001"

    print("\n=== Running Frame Level Auto Calibration ===")
    print(f"Using prefix: {prefix}")
    print(f"Dataset path: {dataset_path}")

    success = run_calibration(prefix, dataset_path)

    if success:
        print("Calibration completed successfully.\n")
    else:
        print("Calibration failed.\n")



# =====================================================================
# Allow direct execution for testing:
# python calibrate_frame_level_offset.py
# =====================================================================
if __name__ == "__main__":
    run()

