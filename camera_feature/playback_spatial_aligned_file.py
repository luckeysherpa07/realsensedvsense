import os
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

# --- 1. SPATIAL ALIGNMENT (Calibration Loading) ---
# Custom loader/constructor logic to handle OpenCV's specific YAML format (!!opencv-matrix)
# This section must be outside the function for correct PyYAML registration.

class OpenCVLoader(yaml.FullLoader):
    """Custom YAML Loader inheriting from FullLoader for OpenCV data."""
    pass

def opencv_matrix_constructor(loader, node):
    """Constructor to convert OpenCV YAML matrix into a NumPy array."""
    mapping = loader.construct_mapping(node, deep=True)
    rows = mapping['rows']
    cols = mapping['cols']
    data = mapping['data']
    # Ensure data is treated as floating point numbers (dt: d)
    return np.array(data, dtype=np.float64).reshape(rows, cols)

# Register the custom constructor with the new loader class
yaml.add_constructor('!opencv-matrix', opencv_matrix_constructor, Loader=OpenCVLoader)
yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor, Loader=OpenCVLoader)

def load_calibration_data(filepath):
    """Loads calibration matrices from the YAML file using the custom loader."""
    
    # Resolve the path to be absolute, starting from the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)
    
    # Since the YAML file is outside the 'camera_feature' directory, adjust the path
    # If the file is 'sibling' to 'camera_feature', we need to go up one level.
    # We use os.path.join and os.pardir to handle this robustly.
    
    # Assumes stereo_calibration_checkerboard.yaml is a sibling of camera_feature/
    calib_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    full_path = os.path.join(calib_dir, "stereo_calibration_checkerboard.yaml")
    
    if not os.path.exists(full_path):
        print(f"Error: Calibration file not found at expected path: {full_path}")
        return None, None, None, None, None, None

    with open(full_path, 'r') as f:
        try:
            # Use the custom OpenCVLoader
            calib_data = yaml.load(f, Loader=OpenCVLoader)
        except Exception as e:
            # We skip the error printing here to prevent confusing duplicate messages
            print(f"Critical YAML loading error: {e}")
            return None, None, None, None, None, None

    # Extract and rename for clarity
    K_dvs = calib_data['DVS_intrinsics']
    D_dvs = calib_data['DVS_distortion'].flatten()
    K_ir = calib_data['IR_intrinsics']
    D_ir = calib_data['IR_distortion'].flatten()
    R_ir_to_dvs = calib_data['Rotation']      # R matrix
    T_ir_to_dvs = calib_data['Translation']  # T vector

    return K_dvs, D_dvs, K_ir, D_ir, R_ir_to_dvs, T_ir_to_dvs


def project_dvs_to_ir(dvs_points, K_dvs, D_dvs, K_ir, D_ir, R_ir_to_dvs, T_ir_to_dvs):
    """
    Projects 2D DVS pixel coordinates onto the IR camera's image plane.
    
    Args:
        dvs_points (np.array): Nx2 array of DVS pixel coordinates (x, y).
        ... (calibration parameters) ...
        
    Returns:
        np.array: Nx2 array of projected IR pixel coordinates.
    """
    
    # 1. Undistort DVS points (normalize coordinates)
    undistorted_dvs_points = cv2.undistortPoints(
        dvs_points.reshape(-1, 1, 2), K_dvs, D_dvs, R=None, P=K_dvs
    )
    
    # 2. Setup the transformation from DVS frame to IR frame
    R_dvs_to_ir = R_ir_to_dvs.T # R is IR -> DVS, so R.T is DVS -> IR
    T_dvs_to_ir = -R_ir_to_dvs.T @ T_ir_to_dvs

    # 3. Prepare 3D points in DVS frame (assuming Z=1 plane)
    points_3d_dvs = np.hstack((undistorted_dvs_points.reshape(-1, 2), np.ones((len(dvs_points), 1))))
    
    # 4. Project 3D points from DVS frame to IR image plane
    ir_pixels, _ = cv2.projectPoints(
        objectPoints=points_3d_dvs,
        rvec=cv2.Rodrigues(R_dvs_to_ir)[0], # Rotation vector
        tvec=T_dvs_to_ir,                     # Translation vector
        cameraMatrix=K_ir,
        distCoeffs=D_ir
    )

    return ir_pixels.reshape(-1, 2)

# --- 2. MAIN PLAYBACK LOGIC (Time and Spatial Alignment) ---

def run():
    # Configuration based on the confirmed hierarchy:
    # 'dataset' and 'stereo_calibration_checkerboard.yaml' are siblings of 'camera_feature'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    
    dataset_path = os.path.join(parent_dir, "dataset") 
    calib_file_placeholder = "stereo_calibration_checkerboard.yaml" # Used internally in load_calibration_data
    
    # --- Load Calibration ---
    K_dvs, D_dvs, K_ir, D_ir, R, T = load_calibration_data(calib_file_placeholder)
    if K_dvs is None:
        print("Failed to load calibration data. Please check YAML syntax and file path.")
        return

    # Extract unique prefixes
    files = os.listdir(dataset_path) if os.path.exists(dataset_path) else []
    prefixes = set()
    for f in files:
        name = os.path.splitext(f)[0]
        prefix = "_".join(name.split("_")[:-1])
        prefixes.add(prefix)
    prefixes = sorted(list(prefixes))

    if not prefixes:
        print(f"No recordings found in dataset directory: {dataset_path}")
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

    # Paths for selected recording
    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # -------------------------------
    # Read calibration offset (Time Alignment)
    # -------------------------------
    delta_seconds = 0.0
    if os.path.exists(delta_file):
        with open(delta_file, "r") as f:
            delta_seconds = float(f.read().strip())
        print(f"Time offset (delta_seconds): {delta_seconds} s")
    else:
        print("Warning: Time delta file not found. Assuming perfect time synchronization (delta_seconds=0).")

    # -------------------------------
    # Event iterator setup
    # -------------------------------
    mv_iterator = None
    event_frame_gen = None
    event_frame = None
    event_start_ts = None
    IR_HEIGHT, IR_WIDTH = 0, 0 # Will be updated with RealSense frame size

    if os.path.exists(event_file):
        mv_iterator = EventsIterator(input_path=event_file, delta_t=10000)
        # Note: DVS resolution (height, width) is derived here, but the event_frame
        # will be generated at IR_HEIGHT x IR_WIDTH after the first RealSense frame is read.
        if not is_live_camera(event_file):
            mv_iterator = LiveReplayEventsIterator(mv_iterator)

        ev_iter = iter(mv_iterator)
    else:
        ev_iter = iter([None])
        print(f"Warning: Event file {event_file} not found.")


    # -------------------------------
    # RealSense bag setup
    # -------------------------------
    pipeline = None
    if os.path.exists(bag_file):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False) 
        
        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"Error starting RealSense pipeline: {e}")
            pipeline = None
        else:
            playback = pipeline_profile.get_device().as_playback()
            playback.set_real_time(False) 
            colorizer = rs.colorizer()
    else:
        print(f"Warning: RealSense bag file {bag_file} not found.")


    print("Press ESC to exit any window.")

    # -------------------------------
    # Playback loop with timestamp and spatial alignment
    # -------------------------------
    start_real_ts = None
    leftover_events = None 

    try:
        while True:
            # ---------------- RealSense frames ----------------
            depth_frame = color_frame = None
            real_ts = None
            if pipeline:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    print("End of RealSense stream.")
                    break
                    
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    if IR_HEIGHT == 0:
                        IR_HEIGHT, IR_WIDTH = color_image.shape[:2]
                        
                        # Initialize frame generator to the target (IR) resolution
                        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=IR_WIDTH,
                                                                           sensor_height=IR_HEIGHT,
                                                                           fps=60,
                                                                           palette=ColorPalette.Dark)
                        # Initialize event frame to the target resolution
                        event_frame = np.zeros((IR_HEIGHT, IR_WIDTH, 3), dtype=np.uint8)
                        
                        def on_cd_frame_cb_warped(ts, cd_frame):
                            nonlocal event_frame
                            # Use decay for visualization
                            alpha = 0.5
                            event_frame = cv2.addWeighted(event_frame, alpha, cd_frame, 1 - alpha, 0)
                            
                        event_frame_gen.set_output_callback(on_cd_frame_cb_warped)


                if depth_frame:
                    depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    real_ts = depth_frame.get_timestamp()
                    if start_real_ts is None:
                        start_real_ts = real_ts

                    t_rs = int((real_ts - start_real_ts) * 1000)

            if pipeline and real_ts is None:
                 continue
                 
            # ---------------- Event accumulation (Time Alignment) ----------------
            last_event_ts = None
            events_in_frame = 0
            accumulated_events = []
            
            if mv_iterator and real_ts is not None:
                
                # 1. Start with leftovers if they exist
                current_chunk = None
                if leftover_events is not None:
                    current_chunk = leftover_events
                    leftover_events = None
                
                # 2. Fetch loop
                while True:
                    # If no chunk available, get a new one
                    if current_chunk is None:
                        try:
                            current_chunk = next(ev_iter)
                        except StopIteration:
                            break
                    
                    if current_chunk is None or current_chunk.size == 0:
                        break

                    if event_start_ts is None:
                        event_start_ts = current_chunk["t"][0]

                    # Calculate aligned timestamps
                    ev_ts_aligned = current_chunk["t"] - event_start_ts + int(delta_seconds * 1e6)

                    # Check against the current RealSense timestamp (t_rs)
                    if ev_ts_aligned[-1] <= t_rs:
                        # Case A: Entire chunk is within the current frame time
                        accumulated_events.append(current_chunk)
                        last_event_ts = ev_ts_aligned[-1]
                        events_in_frame += current_chunk.size
                        current_chunk = None
                    else:
                        # Case B: Chunk crosses the timeline. Split it.
                        mask = ev_ts_aligned <= t_rs
                        
                        # Valid part (current frame)
                        valid_part = current_chunk[mask]
                        if valid_part.size > 0:
                            accumulated_events.append(valid_part)
                            last_event_ts = ev_ts_aligned[mask][-1]
                            events_in_frame += valid_part.size
                        
                        # Future part (save for next frame)
                        leftover_events = current_chunk[~mask]
                        break

            # ---------------- Spatial Alignment & Visualization ----------------
            if accumulated_events and event_frame_gen:
                combined_events = np.concatenate(accumulated_events)
                
                # Extract event pixel coordinates
                dvs_points = np.stack((combined_events['x'], combined_events['y']), axis=1).astype(np.float32)
                
                # 1. Project DVS points to IR image plane
                ir_points = project_dvs_to_ir(dvs_points, K_dvs, D_dvs, K_ir, D_ir, R, T)
                
                # 2. Clip points to IR boundaries and create a new event array
                ir_points = np.round(ir_points).astype(int)
                
                # Check for valid pixels
                x_valid = (ir_points[:, 0] >= 0) & (ir_points[:, 0] < IR_WIDTH)
                y_valid = (ir_points[:, 1] >= 0) & (ir_points[:, 1] < IR_HEIGHT)
                valid_mask = x_valid & y_valid
                
                if np.any(valid_mask):
                    # Create a new event buffer with projected coordinates
                    aligned_events = combined_events[valid_mask].copy()
                    aligned_events['x'] = ir_points[valid_mask, 0].astype(aligned_events['x'].dtype)
                    aligned_events['y'] = ir_points[valid_mask, 1].astype(aligned_events['y'].dtype)

                    # 3. Process the spatially aligned events
                    event_frame_gen.process_events(aligned_events)
                
            # ---------------- Display frames ----------------
            if color_frame is not None:
                overlay = color_image.copy()
                if event_frame is not None:
                    cv2.addWeighted(event_frame, 0.7, overlay, 0.3, 0, overlay)
                cv2.imshow("Aligned RGB + Event Overlay", overlay)

            if depth_frame is not None:
                cv2.imshow("Depth Stream", depth_color)
            
            if event_frame is not None:
                cv2.imshow("Projected Event Stream", event_frame)

            # ---------------- Debug Terminal Output ----------------
            if pipeline and real_ts is not None:
                print(f"RS ms: {real_ts:.3f}, "
                      f"Calibrated RS µs: {t_rs}, "
                      f"Event aligned last ts µs: {last_event_ts if last_event_ts is not None else 'N/A'}, "
                      f"Events projected: {events_in_frame}")

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()