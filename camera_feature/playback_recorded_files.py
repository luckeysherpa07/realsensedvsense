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

    # -------------------------------
    # Event iterator
    # -------------------------------
    mv_iterator = None
    event_frame = None
    if os.path.exists(event_file):
        mv_iterator = EventsIterator(input_path=event_file, delta_t=1000)
        height, width = mv_iterator.get_size()
        if not is_live_camera(event_file):
            mv_iterator = LiveReplayEventsIterator(mv_iterator)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width,
                                                           sensor_height=height,
                                                           fps=25,
                                                           palette=ColorPalette.Dark)
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame
            event_frame = cd_frame.copy()

        event_frame_gen.set_output_callback(on_cd_frame_cb)

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

    try:
        # -------------------------------
        # Main loop
        # -------------------------------
        # Create an iterator for the event camera if available
        if mv_iterator:
            ev_iter = iter(mv_iterator)
        else:
            ev_iter = iter([None])  # dummy iterator

        while True:
            # -------------------------------
            # Event frames
            # -------------------------------
            evs = next(ev_iter, None)
            if mv_iterator and evs is None:
                break  # end of events
            if evs is not None:
                event_frame_gen.process_events(evs)

            # -------------------------------
            # RealSense frames
            # -------------------------------
            if pipeline:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if depth_frame:
                    depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    cv2.imshow("Depth Stream", depth_color)
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imshow("RGB Stream", color_image)

            # -------------------------------
            # Display event frame
            # -------------------------------
            if event_frame is not None:
                cv2.imshow("Event", event_frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
