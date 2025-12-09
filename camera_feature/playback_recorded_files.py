import os
import cv2
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import numpy as np

# -------------------------------
# Main playback function
# -------------------------------
def run():
    dataset_path = "dataset"
    files = os.listdir(dataset_path)

    # Extract unique prefixes
    prefixes = set()
    for f in files:
        name = os.path.splitext(f)[0]
        prefix = name.split("_")[0]
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

    # File paths
    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    ir_file = os.path.join(dataset_path, f"{selected_prefix}_ir.avi")
    depth_file = os.path.join(dataset_path, f"{selected_prefix}_depth.avi")
    rgb_file = os.path.join(dataset_path, f"{selected_prefix}_rgb.avi")

    # -------------------------------
    # Load video captures
    # -------------------------------
    caps = {}
    if os.path.exists(ir_file):
        caps['IR'] = cv2.VideoCapture(ir_file)
    if os.path.exists(depth_file):
        caps['Depth'] = cv2.VideoCapture(depth_file)
    if os.path.exists(rgb_file):
        caps['RGB'] = cv2.VideoCapture(rgb_file)

    # -------------------------------
    # Load event iterator
    # -------------------------------
    if os.path.exists(event_file):
        mv_iterator = EventsIterator(input_path=event_file, delta_t=1000)
        height, width = mv_iterator.get_size()
        if not is_live_camera(event_file):
            mv_iterator = LiveReplayEventsIterator(mv_iterator)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                           fps=25, palette=ColorPalette.Dark)
        # Store current frame
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame
            event_frame = cd_frame.copy()

        event_frame_gen.set_output_callback(on_cd_frame_cb)
    else:
        mv_iterator = None
        event_frame = None

    # -------------------------------
    # Main playback loop
    # -------------------------------
    print("Press ESC to exit any window.")
    for evs in mv_iterator if mv_iterator else [None]:
        EventLoop.poll_and_dispatch()

        # Process event frames
        if mv_iterator and evs is not None:
            event_frame_gen.process_events(evs)

        # Read videos
        frames = {}
        for key, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                # Restart video to loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames[key] = frame

        # Display videos
        for key, frame in frames.items():
            cv2.imshow(key, frame)

        # Display event frame
        if event_frame is not None:
            cv2.imshow('Event', event_frame)

        # Exit check
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if mv_iterator and mv_iterator.should_close() if hasattr(mv_iterator, 'should_close') else False:
            break

    # Release resources
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()