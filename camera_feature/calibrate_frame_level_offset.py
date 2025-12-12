import os
import numpy as np
import cv2
import pyrealsense2 as rs
from metavision_core.event_io import EventsIterator, is_live_camera


# =====================================================================
# MAIN CALIBRATION FUNCTION
# =====================================================================
def run_calibration(prefix, dataset_path):
    """
    Calibrate time offset between event camera and RealSense RGB camera.
    Creates:
      - <prefix>_delta_seconds.txt
      - <prefix>_ev_signal.txt
      - <prefix>_rgb_signal.txt
    inside dataset_path.
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
    # 1. READ EVENT CAM TIME RANGE
    # -----------------------------------------------------
    print("Reading event timestamp range...")

    mv = EventsIterator(event_file, delta_t=1000000)

    ev_start = None
    ev_end = None

    for evs in mv:
        if evs is None or len(evs) == 0:
            continue
        if ev_start is None:
            ev_start = evs["t"][0]
        ev_end = evs["t"][-1]

    if ev_start is None or ev_end is None:
        print("❌ No events found.")
        return False

    ev_start_s = ev_start * 1e-6
    ev_end_s   = ev_end * 1e-6

    print(f"Event time range: {ev_start_s:.3f} → {ev_end_s:.3f} sec")

    # -----------------------------------------------------
    # 2. READ REALSENSE TIME RANGE (RELATIVE)
    # -----------------------------------------------------
    print("Reading RealSense timestamp range...")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    # IMPORTANT: disable real-time playback to avoid dropped frames
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

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

            rel_ts = ts_raw - rs_start_raw
            rs_end_rel = rel_ts

    except Exception:
        pass

    pipeline.stop()

    if rs_end_rel is None:
        print("❌ No RGB frames found in RealSense bag.")
        return False

    print(f"RGB time range (relative): 0.000 → {rs_end_rel:.3f} sec")

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
    fs = 1000  # 1 kHz sampling resolution
    L = int((overlap_end - overlap_start) * fs)

    ev_signal  = np.zeros(L)
    rgb_signal = np.zeros(L)

    # -----------------------------------------------------
    # 5. BUILD EVENT SIGNAL
    # -----------------------------------------------------
    print("Building 1-D event signal...")

    mv = EventsIterator(event_file, delta_t=1000000)

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

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    prev_gray = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                break

            ts_raw = c.get_timestamp() * 0.001
            rel_ts = ts_raw - rs_start_raw

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
    # 7. CROSS-CORRELATION (NORMALIZED)
    # -----------------------------------------------------
    print("Computing cross-correlation...")

    a = ev_signal - ev_signal.mean()
    b = rgb_signal - rgb_signal.mean()

    if np.std(a) == 0 or np.std(b) == 0:
        print("❌ One of the signals is zero. Cannot correlate.")
        return False

    a /= np.std(a)
    b /= np.std(b)

    corr = np.correlate(a, b, mode="full")
    lag = corr.argmax() - (L - 1)

    delta_seconds = lag / fs

    print(f"\nEstimated Δt = {delta_seconds:.6f} seconds")
    print("Positive Δt → EVENT stream is ahead of RGB.")

    # -----------------------------------------------------
    # 8. SAVE DELTA TIME
    # -----------------------------------------------------
    out_file = os.path.join(dataset_path, f"{prefix}_delta_seconds.txt")
    with open(out_file, "w") as f:
        f.write(f"{delta_seconds:.9f}\n")

    print(f"Saved calibration to: {out_file}")

    # -----------------------------------------------------
    # 9. SAVE SIGNALS (OPTION 1)
    # -----------------------------------------------------
    print("Saving EV/RGB signals for ChatGPT plotting...")

    ev_file = os.path.join(dataset_path, f"{prefix}_ev_signal.txt")
    rgb_file = os.path.join(dataset_path, f"{prefix}_rgb_signal.txt")

    np.savetxt(ev_file, ev_signal)
    np.savetxt(rgb_file, rgb_signal)

    print(f"Saved event signal: {ev_file}")
    print(f"Saved RGB signal:   {rgb_file}")
    print("Upload these files here to generate the real cross-correlation plot.\n")

    print("===============================================\n")

    return True


# =====================================================================
# WRAPPER FUNCTION
# =====================================================================
def run():
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
# MAIN ENTRY
# =====================================================================
if __name__ == "__main__":
    run()
