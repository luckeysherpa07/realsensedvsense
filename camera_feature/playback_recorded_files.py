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

    # Extract unique prefixes based on _event.raw convention
    prefixes = set()
    for f in files:
        if f.endswith("_event.raw"):
            prefix = f.replace("_event.raw", "")
            prefixes.add(prefix)
    prefixes = sorted(list(prefixes))

    if not prefixes:
        print("No recordings found in dataset (looking for *_event.raw).")
        return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1):
        print(f"{idx}. {prefix}")

    choice = input(f"\nSelect a recording to play (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(prefixes):
        print("Invalid choice!")
        return

    selected_prefix = prefixes[int(choice) - 1]
    print(f"\nPlaying recording: {selected_prefix}")

    # Paths
    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # -------------------------------
    # 1. Read calibration offset
    # -------------------------------
    delta_us = 0
    if os.path.exists(delta_file):
        try:
            with open(delta_file, "r") as f:
                val = float(f.read().strip())
                delta_us = int(val * 1e6)
            print(f"Calibration offset loaded: {delta_us / 1e6} s")
        except ValueError:
            print("Warning: Could not parse delta file.")

    # -------------------------------
    # 2. Event iterator setup
    # -------------------------------
    mv_iterator = None
    event_frame_gen = None
    event_frame = None
    
    # Buffers for event logic
    leftover_events = None 
    ev_start_ts_us = None
    
    if os.path.exists(event_file):
        # We use a smaller delta_t for the iterator to allow fine-grained seeking
        mv_iterator = EventsIterator(input_path=event_file, delta_t=1000)
        height, width = mv_iterator.get_size()
        
        # Generate frames at 60 FPS or based on input flow
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=60, 
            palette=ColorPalette.Dark
        )
        
        # Initialize black frame
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame
            event_frame = cd_frame.copy()

        event_frame_gen.set_output_callback(on_cd_frame_cb)
        ev_iter = iter(mv_iterator)
    else:
        print(f"Error: Event file not found at {event_file}")
        return

    # -------------------------------
    # 3. RealSense bag setup
    # -------------------------------
    pipeline = None
    if os.path.exists(bag_file):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False)
        
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        
        # CRITICAL: Set real_time=True to view at normal speed
        # Set to False if you want to process as fast as possible (fast forward)
        playback.set_real_time(True) 
        
        colorizer = rs.colorizer()
    else:
        print("Error: Bag file not found.")
        return

    print("Controls: Press 'ESC' to exit.")

    # -------------------------------
    # 4. Playback Loop
    # -------------------------------
    rs_start_ts_ms = None

    try:
        while True:
            # A. Get RealSense Data
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("RealSense playback finished.")
                break

            depth_frame = frames.get_depth_frame()
            # Try to get IR frame from index 1 (usually Left IR), fallback to 0
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                ir_frame = frames.get_infrared_frame(0)

            # Skip if frames are incomplete
            if not depth_frame and not ir_frame:
                continue

            # Handle RS Timing
            # get_timestamp() returns time in milliseconds
            current_rs_ts_ms = frames.get_timestamp() 
            
            if rs_start_ts_ms is None:
                rs_start_ts_ms = current_rs_ts_ms

            # Time elapsed since start of RS video (in microseconds)
            elapsed_time_us = int((current_rs_ts_ms - rs_start_ts_ms) * 1000)

            # --- Prepare Images ---
            depth_color = None
            ir_bgr = None

            if depth_frame:
                depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if ir_frame:
                ir_image_raw = np.asanyarray(ir_frame.get_data())
                if ir_image_raw.dtype == np.uint16:
                    ir_image_8bit = cv2.normalize(ir_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    ir_image_8bit = ir_image_raw.astype(np.uint8)
                ir_bgr = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)

            # B. Process Events up to this timestamp
            if ev_iter:
                # 1. Initialize Event Start Timestamp if needed
                if ev_start_ts_us is None:
                    try:
                        # Grab first batch to determine start time
                        if leftover_events is None:
                            leftover_events = next(ev_iter)
                        if len(leftover_events) > 0:
                            ev_start_ts_us = leftover_events['t'][0]
                        else:
                            # If empty, keep trying next batches
                            while len(leftover_events) == 0:
                                leftover_events = next(ev_iter)
                            ev_start_ts_us = leftover_events['t'][0]
                    except StopIteration:
                        print("Event file is empty.")
                        break

                # 2. Calculate Target Timestamp
                # target = Start_Event_Time + Duration_Elapsed_In_RS - Offset
                target_ev_raw = ev_start_ts_us + elapsed_time_us - delta_us

                accumulated = []
                
                # Add leftovers from previous loop
                current_max_t = 0
                if leftover_events is not None and len(leftover_events) > 0:
                    accumulated.append(leftover_events)
                    current_max_t = leftover_events['t'][-1]
                    leftover_events = None # Consumed

                # Fetch new events until we pass the target timestamp
                while current_max_t < target_ev_raw:
                    try:
                        new_batch = next(ev_iter)
                        if len(new_batch) == 0: 
                            continue
                        accumulated.append(new_batch)
                        current_max_t = new_batch['t'][-1]
                    except StopIteration:
                        # Event file finished before RS file
                        break
                
                # Process accumulated events
                if accumulated:
                    all_evs = np.concatenate(accumulated)
                    
                    # Find split point: events <= target_ev_raw
                    split_idx = np.searchsorted(all_evs['t'], target_ev_raw, side='right')
                    
                    events_to_process = all_evs[:split_idx]
                    leftover_events = all_evs[split_idx:] # Save for next frame
                    
                    if len(events_to_process) > 0:
                        # Update the generator state
                        event_frame_gen.process_events(events_to_process)

            # C. Visualization
            if event_frame is not None:
                # ---------------- Window 1: Event Stream ----------------
                cv2.imshow("Event Stream", event_frame)

                # Helper function to overlay
                def overlay_images(bg_img, overlay_img, alpha=0.6):
                    if bg_img is None or overlay_img is None:
                        return None
                    h, w = bg_img.shape[:2]
                    # Resize overlay to match background
                    if overlay_img.shape[:2] != (h, w):
                        overlay_resized = cv2.resize(overlay_img, (w, h))
                    else:
                        overlay_resized = overlay_img
                    
                    # Blend
                    beta = 1.0 - alpha
                    return cv2.addWeighted(overlay_resized, alpha, bg_img, beta, 0)

                # ---------------- Window 2: IR + Event Overlay ----------------
                if ir_bgr is not None:
                    vis_ir = overlay_images(ir_bgr, event_frame, alpha=0.5)
                    cv2.imshow("IR + Event Overlay", vis_ir)

                # ---------------- Window 3: Depth + Event Overlay ----------------
                if depth_color is not None:
                    vis_depth = overlay_images(depth_color, event_frame, alpha=0.5)
                    cv2.imshow("Depth + Event Overlay", vis_depth)

            key = cv2.waitKey(1)
            if key == 27: # ESC
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Finished.")

if __name__ == "__main__":
    run()