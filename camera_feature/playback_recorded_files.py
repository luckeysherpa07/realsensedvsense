import os
import cv2
import numpy as np
import pyrealsense2 as rs
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

def run():
    dataset_path = "dataset"
    files = os.listdir(dataset_path)

    # Extract unique prefixes
    prefixes = set()
    for f in files:
        name = os.path.splitext(f)[0]
        prefix = "_".join(name.split("_")[:-1])
        prefixes.add(prefix)
    prefixes = sorted(list(prefixes))

    if not prefixes:
        print("No recordings found in dataset.")
        return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1):
        print(f"{idx}. {prefix}")

    choice = input(f"\nSelect a recording to play (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(prefixes):
        print("Invalid choice!")
        return

    selected_prefix = prefixes[int(choice) - 1]
    print(f"\nPlaying recording: {selected_prefix}\n")

    # Paths
    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # -------------------------------
    # Read calibration offset
    # -------------------------------
    delta_seconds = 0.0
    if os.path.exists(delta_file):
        with open(delta_file, "r") as f:
            delta_seconds = float(f.read().strip())
        print(f"Calibration offset (delta_seconds) read from file: {delta_seconds} s")
        print("Sign convention: Δt = lag from cross-correlation (event vs RGB)")
        if delta_seconds > 0:
            print("Event stream lags RGB; shift events earlier to align.")
        else:
            print("Event stream leads RGB; shift events later to align.")

    # -------------------------------
    # Event iterator setup
    # -------------------------------
    mv_iterator = None
    event_frame = None
    event_start_ts = None
    if os.path.exists(event_file):
        # CHANGED: delta_t from 1000 to 10000 (10ms) to reduce loop overhead
        mv_iterator = EventsIterator(input_path=event_file, delta_t=10000)
        height, width = mv_iterator.get_size()
        if not is_live_camera(event_file):
            mv_iterator = LiveReplayEventsIterator(mv_iterator)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width,
                                                           sensor_height=height,
                                                           fps=60,
                                                           palette=ColorPalette.Dark)
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame
            # Use decay to avoid accumulation lag
            alpha = 0.5
            event_frame = cv2.addWeighted(event_frame, alpha, cd_frame, 1 - alpha, 0)

        event_frame_gen.set_output_callback(on_cd_frame_cb)
        ev_iter = iter(mv_iterator)
    else:
        ev_iter = iter([None])

    # -------------------------------
    # RealSense bag setup
    # -------------------------------
    # -------------------------------
    # RealSense bag setup
    # -------------------------------
    pipeline = None
    if os.path.exists(bag_file):
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Disable repeat to avoid confusion at end of file
        config.enable_device_from_file(bag_file, repeat_playback=False) 
        
        pipeline_profile = pipeline.start(config)

        # --- CRITICAL FIX: Disable Real-Time Playback ---
        # This forces the bag to play frame-by-frame, waiting for your processing
        playback = pipeline_profile.get_device().as_playback()
        playback.set_real_time(False) 
        # ------------------------------------------------
        
        colorizer = rs.colorizer()

    print("Press ESC to exit any window.")

    # -------------------------------
    # Playback loop with timestamp alignment
    # -------------------------------
    start_real_ts = None  # first RealSense timestamp (ms)
    frame_delay_us = int(1e6 / 60)  # event frame generation delay (~16.6ms for 60Hz)

    try:
        while True:
            # ---------------- RealSense frames ----------------
            depth_frame = color_frame = None
            real_ts = None
            if pipeline:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if depth_frame:
                    depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    real_ts = depth_frame.get_timestamp()  # ms
                    if start_real_ts is None:
                        start_real_ts = real_ts

                    # Convert RealSense timestamp to µs relative to start
                    t_rs = int((real_ts - start_real_ts) * 1000)

                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())

            # ---------------- Event frames ----------------
            last_event_ts = None
            events_in_frame = 0
            if mv_iterator:
                accumulated_events = []
                while True:
                    try:
                        evs = next(ev_iter)
                    except StopIteration:
                        evs = None

                    if evs is None or evs.size == 0:
                        break

                    if event_start_ts is None:
                        event_start_ts = evs["t"][0]

                    # Align event timestamps relative to first event and apply delta_seconds
                    ev_ts_aligned = evs["t"] - event_start_ts + int(delta_seconds * 1e6)

                    # Accumulate only events up to current RS timestamp
                    if pipeline and real_ts is not None:
                        mask = ev_ts_aligned <= t_rs
                        if np.any(mask):
                            accumulated_events.append(evs[mask])
                            last_event_ts = ev_ts_aligned[mask][-1]
                            events_in_frame += mask.sum()
                        if ev_ts_aligned[-1] > t_rs:
                            break
                    else:
                        accumulated_events.append(evs)
                        last_event_ts = ev_ts_aligned[-1]
                        events_in_frame += evs.size

                if accumulated_events:
                    combined = np.concatenate(accumulated_events)
                    event_frame_gen.process_events(combined)

            # ---------------- Display frames ----------------
            if color_frame is not None:
                overlay = color_image.copy()
                if event_frame is not None:
                    cv2.addWeighted(event_frame, 0.7, overlay, 0.3, 0, overlay)
                cv2.imshow("RGB + Event Overlay", overlay)

            if depth_frame is not None:
                cv2.imshow("Depth Stream", depth_color)
            if event_frame is not None:
                cv2.imshow("Event Stream", event_frame)

            # ---------------- Debug Terminal Output ----------------
            if pipeline and real_ts is not None:
                print(f"RS ms: {real_ts:.3f}, "
                      f"Calibrated RS µs: {t_rs}, "
                      f"Event aligned last ts µs: {last_event_ts if last_event_ts is not None else 'N/A'}, "
                      f"Events in frame: {events_in_frame}")

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
