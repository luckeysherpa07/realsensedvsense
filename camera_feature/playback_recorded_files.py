import os
import cv2
import numpy as np
import pyrealsense2 as rs
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

def run():
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    files = os.listdir(dataset_path)

    # Extract unique prefixes
    prefixes = set()
    for f in files:
        name = os.path.splitext(f)[0]
        # logic to handle filenames properly
        if "_" in name:
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
    # 1. Read calibration offset
    # -------------------------------
    delta_us = 0
    if os.path.exists(delta_file):
        with open(delta_file, "r") as f:
            val = float(f.read().strip())
            delta_us = int(val * 1e6)
        print(f"Calibration offset: {delta_us / 1e6} s")
    
    # -------------------------------
    # 2. Event iterator setup
    # -------------------------------
    mv_iterator = None
    event_frame_gen = None
    event_frame = None
    
    # We need a buffer to hold events read from disk that represent time > current_rs_frame
    leftover_events = None 
    
    if os.path.exists(event_file):
        # NOTE: Do NOT use LiveReplayEventsIterator here. We want to read 
        # as fast as possible to sync with the Bag file, not simulate wall-clock time.
        mv_iterator = EventsIterator(input_path=event_file, delta_t=10000)
        height, width = mv_iterator.get_size()
        
        # Generator for visualization
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=60, # Target FPS for the event view
            palette=ColorPalette.Dark
        )
        
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame
            event_frame = cd_frame.copy()

        event_frame_gen.set_output_callback(on_cd_frame_cb)
        ev_iter = iter(mv_iterator)
    else:
        print(f"Warning: Event file not found at {event_file}")
        ev_iter = None

    # -------------------------------
    # 3. RealSense bag setup
    # -------------------------------
    pipeline = None
    if os.path.exists(bag_file):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False)
        
        profile = pipeline.start(config)
        
        # CRITICAL: Disable real-time playback.
        # This allows us to request frames sequentially without dropping them,
        # ensuring we can draw the exact events corresponding to that frame.
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        
        colorizer = rs.colorizer()
    else:
        print("Error: Bag file not found.")
        return

    print("Press ESC to exit.")

    # -------------------------------
    # 4. Playback Loop
    # -------------------------------
    
    # Timestamps
    rs_start_ts_ms = None  # The timestamp of the first RS frame
    ev_start_ts_us = None  # The timestamp of the first Event

    try:
        while True:
            # A. Get RealSense Data
            frames = pipeline.wait_for_frames()
            
            # Handle End of File (RealSense wraps or stops)
            # Note: rs2 doesn't easily signal EOF in blocking mode, usually throws or hangs.
            # We assume continuous stream until user quits or data runs out.
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Handle RS Timing
            current_rs_ts_ms = depth_frame.get_timestamp()
            if rs_start_ts_ms is None:
                rs_start_ts_ms = current_rs_ts_ms

            # Calculate how much time has passed in the video (in microseconds)
            elapsed_time_us = int((current_rs_ts_ms - rs_start_ts_ms) * 1000)

            # Process Images
            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # B. Process Events up to this timestamp
            if ev_iter:
                # 1. Initialize start timestamp if needed
                if ev_start_ts_us is None:
                    # Peek at first batch to get start time, or use leftover
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]
                    else:
                        try:
                            # Fetch one batch to init time
                            init_evs = next(ev_iter)
                            ev_start_ts_us = init_evs['t'][0]
                            leftover_events = init_evs
                        except StopIteration:
                            pass # Empty file

                if ev_start_ts_us is not None:
                    # Calculate the Target Event Timestamp
                    # Target = Start_Event_Time + Elapsed_RS_Time - Calibration_Offset
                    # (Sign of delta depends on your specific calibration file convention)
                    # Based on your snippet: "Event stream lags RGB (delta > 0)" implies 
                    # we must subtract delta to align? Or add? 
                    # Your code did: ev_ts_aligned = evs["t"] - event_start_ts + int(delta_seconds * 1e6)
                    # Let's align strictly to the elapsed time logic:
                    # We want events where: (ev_t - ev_start) + delta_us <= elapsed_time_us
                    
                    # Therefore, we want raw event timestamp <= :
                    target_ev_raw = ev_start_ts_us + elapsed_time_us - delta_us

                    accumulated = []
                    
                    # 2. Add leftovers from previous iteration
                    if leftover_events is not None and len(leftover_events) > 0:
                        accumulated.append(leftover_events)
                        leftover_events = None

                    # 3. Fetch new events until we pass the target time
                    chunk_max_t = 0
                    if len(accumulated) > 0:
                        chunk_max_t = accumulated[-1]['t'][-1]

                    while chunk_max_t < target_ev_raw:
                        try:
                            new_batch = next(ev_iter)
                            if len(new_batch) == 0: 
                                continue
                            accumulated.append(new_batch)
                            chunk_max_t = new_batch['t'][-1]
                        except StopIteration:
                            break
                    
                    # 4. Filter and Split
                    # We now have a list of batches that likely overshot target_ev_raw.
                    # We need to concatenate, split at target, and save the rest as leftover.
                    if accumulated:
                        all_evs = np.concatenate(accumulated)
                        
                        # Find the index where timestamp > target_ev_raw
                        # searchsorted is very fast for sorted arrays (which event streams are)
                        split_idx = np.searchsorted(all_evs['t'], target_ev_raw, side='right')
                        
                        events_to_process = all_evs[:split_idx]
                        leftover_events = all_evs[split_idx:]
                        
                        # Update the frame generator
                        if len(events_to_process) > 0:
                            event_frame_gen.process_events(events_to_process)

            # C. Visualization
            if color_image is not None:
                overlay = color_image.copy()
                
                # Resize event frame to match RGB if necessary
                if event_frame is not None:
                    if event_frame.shape[:2] != overlay.shape[:2]:
                        ev_vis = cv2.resize(event_frame, (overlay.shape[1], overlay.shape[0]))
                    else:
                        ev_vis = event_frame
                    
                    # Create a blended overlay
                    # If event pixel is black, keep RGB. If event is colored, blend.
                    mask = np.any(ev_vis > 0, axis=2).astype(np.uint8)
                    
                    # Standard addWeighted for the whole image
                    cv2.addWeighted(ev_vis, 0.6, overlay, 0.4, 0, overlay)
                    
                    cv2.imshow("Event Stream", ev_vis)

                cv2.imshow("RGB + Event Overlay", overlay)
                cv2.imshow("Depth Stream", depth_color)

            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()