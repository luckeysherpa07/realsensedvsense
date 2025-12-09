import os
import cv2
import numpy as np
import pyrealsense2 as rs
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
import time

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

    # -------------------------------
    # Event iterator setup
    # -------------------------------
    mv_iterator = None
    event_frame = None
    event_start_ts = None
    if os.path.exists(event_file):
        mv_iterator = EventsIterator(input_path=event_file, delta_t=1000)
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
            event_frame = cd_frame.copy()

        event_frame_gen.set_output_callback(on_cd_frame_cb)
        ev_iter = iter(mv_iterator)
    else:
        ev_iter = iter([None])

    # -------------------------------
    # RealSense bag setup
    # -------------------------------
    pipeline = None
    if os.path.exists(bag_file):
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file)
        pipeline_profile = pipeline.start(config)
        colorizer = rs.colorizer()

    print("Press ESC to exit any window.")

    # -------------------------------
    # Playback loop with timestamp alignment and diagnostics
    # -------------------------------
    start_time_wall = time.time()
    start_real_ts = None

    event_frame_count = 0
    real_frame_count = 0
    last_diag_time = time.time()

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
                    aligned_real_ts_us = int((real_ts - start_real_ts) * 1000)
                    print("alignedRealTS..... REAL_TS...... START_REAL_TS", aligned_real_ts_us, real_ts, start_real_ts)
                    real_frame_count += 1
                else:
                    aligned_real_ts_us = None

                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())

            # ---------------- Event frames ----------------
            if mv_iterator:
                accumulated_events = []
                while True:
                    evs = next(ev_iter, None)
                    if evs is None or evs.size == 0:
                        break
                    if event_start_ts is None:
                        event_start_ts = evs["t"][0]

                    ev_ts_aligned = evs["t"] - event_start_ts
                    print("EV_TS_ALINGED...........EVS().......EV_START_TS", ev_ts_aligned, evs['t'], event_start_ts)

                    # Accumulate only events up to current RealSense timestamp
                    if aligned_real_ts_us is not None:
                        mask = ev_ts_aligned <= aligned_real_ts_us
                        if np.any(mask):
                            accumulated_events.append(evs[mask])
                        # Stop accumulating beyond current frame
                        if ev_ts_aligned[-1] > aligned_real_ts_us:
                            break
                    else:
                        accumulated_events.append(evs)

                # Combine all accumulated events for this frame
                if accumulated_events:
                    combined = np.concatenate(accumulated_events)
                    event_frame_gen.process_events(combined)


            # ---------------- Display frames ----------------
            if color_frame is not None:
                cv2.imshow("RGB Stream", color_image)
                print("RGB FRAME #################", color_image)
            if depth_frame is not None:
                cv2.imshow("Depth Stream", depth_color)
            if event_frame is not None:
                cv2.imshow("Event", event_frame)
                print("EVENT FRAME##################", event_frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
