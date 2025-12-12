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
    print(f"\nPlaying recording with IR Overlay: {selected_prefix}\n")

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
    
    leftover_events = None 
    
    if os.path.exists(event_file):
        mv_iterator = EventsIterator(input_path=event_file, delta_t=10000)
        height, width = mv_iterator.get_size()
        
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=60, 
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
        
        # Disable real-time playback for frame-by-frame sync
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
    rs_start_ts_ms = None
    ev_start_ts_us = None

    try:
        while True:
            # A. Get RealSense Data
            frames = pipeline.wait_for_frames()
            
            depth_frame = frames.get_depth_frame()
            
            # --- CHANGED: Get IR Frame instead of Color ---
            # Try getting Left IR (index 1), fallback to index 0
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                ir_frame = frames.get_infrared_frame(0)

            if not depth_frame or not ir_frame:
                continue

            # Handle RS Timing
            current_rs_ts_ms = depth_frame.get_timestamp()
            if rs_start_ts_ms is None:
                rs_start_ts_ms = current_rs_ts_ms

            elapsed_time_us = int((current_rs_ts_ms - rs_start_ts_ms) * 1000)

            # Process Images
            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # Process IR Image
            ir_image_raw = np.asanyarray(ir_frame.get_data())
            
            # Normalize IR if it is 16-bit to make it visible
            if ir_image_raw.dtype == np.uint16:
                ir_image_8bit = cv2.normalize(ir_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                ir_image_8bit = ir_image_raw.astype(np.uint8)

            # Convert Grayscale IR to BGR (3 channels) so we can overlay colored events
            ir_bgr = cv2.cvtColor(ir_image_8bit, cv2.COLOR_GRAY2BGR)

            # B. Process Events up to this timestamp
            if ev_iter:
                if ev_start_ts_us is None:
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]
                    else:
                        try:
                            init_evs = next(ev_iter)
                            ev_start_ts_us = init_evs['t'][0]
                            leftover_events = init_evs
                        except StopIteration:
                            pass 

                if ev_start_ts_us is not None:
                    target_ev_raw = ev_start_ts_us + elapsed_time_us - delta_us

                    accumulated = []
                    
                    if leftover_events is not None and len(leftover_events) > 0:
                        accumulated.append(leftover_events)
                        leftover_events = None

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
                    
                    if accumulated:
                        all_evs = np.concatenate(accumulated)
                        split_idx = np.searchsorted(all_evs['t'], target_ev_raw, side='right')
                        
                        events_to_process = all_evs[:split_idx]
                        leftover_events = all_evs[split_idx:]
                        
                        if len(events_to_process) > 0:
                            event_frame_gen.process_events(events_to_process)

            # C. Visualization (IR Overlay)
            if ir_bgr is not None:
                overlay = ir_bgr.copy()
                
                if event_frame is not None:
                    # Resize event frame to match IR if necessary
                    if event_frame.shape[:2] != overlay.shape[:2]:
                        ev_vis = cv2.resize(event_frame, (overlay.shape[1], overlay.shape[0]))
                    else:
                        ev_vis = event_frame
                    
                    # Create the overlay
                    cv2.addWeighted(ev_vis, 0.7, overlay, 0.5, 0, overlay)
                    
                    cv2.imshow("Event Stream", ev_vis)

                cv2.imshow("IR + Event Overlay", overlay)
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