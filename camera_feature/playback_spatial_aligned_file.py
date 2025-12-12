import os
import cv2
import numpy as np
import pyrealsense2 as rs
import re
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

# -----------------------------------------------------------------------------
# 1. Calibration Parsing Helper
# -----------------------------------------------------------------------------
def load_opencv_yaml(filepath):
    """
    Parses a specific OpenCV YAML format containing matrices without 
    requiring the full opencv yaml parser or external yaml libraries.
    """
    if not os.path.exists(filepath):
        print(f"Calibration file not found: {filepath}")
        return None

    data = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to find matrix definitions
    pattern = re.compile(
        r"(\w+):\s*!!opencv-matrix\s*"
        r"rows:\s*(\d+)\s*"
        r"cols:\s*(\d+)\s*"
        r"dt:\s*[a-z]\s*"
        r"data:\s*\[(.*?)\]", 
        re.DOTALL
    )

    matches = pattern.findall(content)
    for name, rows, cols, raw_data in matches:
        cleaned_data = raw_data.replace('\n', '').strip()
        values = [float(x) for x in cleaned_data.split(',')]
        matrix = np.array(values, dtype=np.float64).reshape(int(rows), int(cols))
        data[name] = matrix

    return data

# -----------------------------------------------------------------------------
# 2. Main Logic
# -----------------------------------------------------------------------------
def run():
    # --- Paths ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "../dataset"))
    calib_file = os.path.abspath(os.path.join(current_dir, "../stereo_calibration_checkerboard.yaml"))

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    # --- Load Calibration ---
    calib_data = load_opencv_yaml(calib_file)
    has_calib = False
    
    # Camera 1 = RealSense (Left/Target)
    # Camera 2 = DVS (Right/Source)
    K_dvs, D_dvs = None, None
    K_ir, D_ir = None, None
    R_stereo = None 

    if calib_data:
        try:
            K_dvs = calib_data['DVS_intrinsics']
            D_dvs = calib_data['DVS_distortion']
            K_ir = calib_data['IR_intrinsics']
            D_ir = calib_data['IR_distortion']
            R_stereo = calib_data['Rotation'] # Rotation from Left to Right
            
            print(f"Calibration loaded successfully.")
            has_calib = True
        except KeyError as e:
            print(f"Missing key in calibration file: {e}")
    else:
        print(f"Warning: Calibration file not found at {calib_file}")

    # --- File Selection ---
    files = os.listdir(dataset_path)
    prefixes = set()
    for f in files:
        name = os.path.splitext(f)[0]
        if "_event" in name:
            prefix = name.replace("_event", "")
            prefixes.add(prefix)
    
    prefixes = sorted(list(prefixes))
    if not prefixes:
        print(f"No event recordings found in {dataset_path}")
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

    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # --- Load Time Offset ---
    delta_us = 0
    if os.path.exists(delta_file):
        with open(delta_file, "r") as f:
            try:
                val = float(f.read().strip())
                delta_us = int(val * 1e6)
                print(f"Loaded Calibration offset: {val:.6f} s")
            except: pass

    # --- Setup Event Stream ---
    if not os.path.exists(event_file):
        print(f"Error: Event file missing: {event_file}")
        return

    mv_iterator = EventsIterator(input_path=event_file, delta_t=10000)
    ev_height, ev_width = mv_iterator.get_size()
    
    # *** CRITICAL FIX: Convert iterable to iterator ***
    ev_it = iter(mv_iterator)

    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=ev_width,
        sensor_height=ev_height,
        fps=100, 
        palette=ColorPalette.Dark
    )
    
    event_frame = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)
    def on_cd_frame_cb(ts, cd_frame):
        nonlocal event_frame
        event_frame = cd_frame.copy()
    event_frame_gen.set_output_callback(on_cd_frame_cb)
    
    event_buffer = np.empty((0,), dtype=[('x', 'u2'), ('y', 'u2'), ('p', 'i2'), ('t', 'i8')])

    # --- Setup RealSense ---
    if not os.path.exists(bag_file):
        print(f"Error: Bag file missing: {bag_file}")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    colorizer = rs.colorizer()

    # --- Mapping Initialization (Lazy Load) ---
    map1, map2 = None, None

    print("\nControls: [ESC] Exit | [SPACE] Pause")

    rs_start_ts_ms = None
    ev_start_ts_us = None
    prev_rs_ts_ms = 0

    try:
        while True:
            # 1. Get RS Frames
            success, frames = pipeline.try_wait_for_frames(timeout_ms=1000)
            if not success: 
                print("End of RealSense recording.")
                break
            
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1) # Try Left IR first
            if not ir_frame: ir_frame = frames.get_infrared_frame(0)

            if not depth_frame or not ir_frame: continue

            # 2. Timing
            current_rs_ts_ms = depth_frame.get_timestamp()
            if rs_start_ts_ms is None:
                rs_start_ts_ms = current_rs_ts_ms
                prev_rs_ts_ms = current_rs_ts_ms
            
            elapsed_time_us = int((current_rs_ts_ms - rs_start_ts_ms) * 1000)

            # 3. Process Events
            if ev_start_ts_us is None:
                try:
                    # Use the iterator 'ev_it'
                    first = next(ev_it)
                    if len(first) > 0:
                        ev_start_ts_us = first['t'][0]
                        event_buffer = first
                except StopIteration: 
                    break

            if ev_start_ts_us is not None:
                target_ev_t = ev_start_ts_us + elapsed_time_us - delta_us
                
                # Fetch more events if needed
                buffer_max = event_buffer['t'][-1] if len(event_buffer) > 0 else 0
                while buffer_max < target_ev_t:
                    try:
                        new_batch = next(ev_it)
                        if len(new_batch) > 0:
                            event_buffer = np.concatenate((event_buffer, new_batch))
                            buffer_max = new_batch['t'][-1]
                        else: break
                    except StopIteration: break
                
                # Process relevant chunk
                if len(event_buffer) > 0:
                    split_idx = np.searchsorted(event_buffer['t'], target_ev_t, side='right')
                    to_process = event_buffer[:split_idx]
                    event_buffer = event_buffer[split_idx:]
                    if len(to_process) > 0:
                        event_frame_gen.process_events(to_process)

            # 4. Prepare Images
            # --- Depth ---
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # --- IR ---
            ir_raw = np.asanyarray(ir_frame.get_data())
            if ir_raw.dtype == np.uint16:
                 ir_8bit = cv2.normalize(ir_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                 ir_8bit = ir_raw.astype(np.uint8)
            
            # Undistort IR (Left) so it matches the pinhole model K_ir
            if has_calib:
                ir_display = cv2.undistort(ir_8bit, K_ir, D_ir)
            else:
                ir_display = ir_8bit

            ir_bgr = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)

            # --- Events & Alignment ---
            if event_frame is not None:
                
                # Init Mapping Once (needs IR resolution)
                if has_calib and map1 is None:
                    ir_h, ir_w = ir_display.shape[:2]
                    
                    # LOGIC:
                    # We are mapping FROM DVS (Source) TO IR (Target).
                    # 'K_dvs', 'D_dvs' are the source intrinsics.
                    # 'K_ir' is the new camera matrix (we want DVS to look like IR).
                    # 'R': We need the rotation that aligns the DVS axes to the IR axes.
                    # R_stereo is Left->Right. So DVS = R * IR + T.
                    # To align DVS to IR, we need Inverse Rotation (Transpose).
                    R_align = R_stereo.T

                    # This creates a lookup map to warp DVS pixels to the IR viewpoint
                    map1, map2 = cv2.initUndistortRectifyMap(
                        K_dvs, D_dvs, R_align, K_ir, (ir_w, ir_h), cv2.CV_16SC2
                    )
                    print(f"Alignment Map Generated. Output Size: {ir_w}x{ir_h}")

                # If calibration exists, Remap events to overlay
                if has_calib and map1 is not None:
                    aligned_events = cv2.remap(event_frame, map1, map2, cv2.INTER_LINEAR)
                else:
                    # Fallback
                    aligned_events = cv2.resize(event_frame, (ir_bgr.shape[1], ir_bgr.shape[0]))

                # --- Visualizations ---
                mask = np.any(aligned_events > 0, axis=2)

                # 1. IR + Events
                overlay_ir = ir_bgr.copy()
                # Blend only where events exist
                overlay_ir[mask] = cv2.addWeighted(ir_bgr, 0.5, aligned_events, 0.8, 0)[mask]
                cv2.imshow("Aligned: IR (L) + Events (R)", overlay_ir)

                # 2. Depth + Events
                # Depth is naturally aligned with Left IR in RS435/455
                overlay_depth = depth_image.copy()
                overlay_depth[mask] = cv2.addWeighted(depth_image, 0.5, aligned_events, 0.8, 0)[mask]
                cv2.imshow("Aligned: Depth (L) + Events (R)", overlay_depth)

                # 3. Raw Events (Reference)
                cv2.imshow("Raw Events", event_frame)

            # 5. Playback Control
            dt = current_rs_ts_ms - prev_rs_ts_ms
            prev_rs_ts_ms = current_rs_ts_ms
            
            key = cv2.waitKey(max(1, int(dt)))
            if key == 27: break
            if key == 32: cv2.waitKey(0)

    finally:
        if 'pipeline' in locals():
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()