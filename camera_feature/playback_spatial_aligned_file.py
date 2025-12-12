import os
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

# --- 1. SPATIAL ALIGNMENT (Calibration Loading) ---

def load_calibration_data(filepath):
    """Loads OpenCV matrices from the YAML file."""
    with open(filepath, 'r') as f:
        # Custom loader to handle OpenCV matrix structure
        def opencv_matrix_constructor(loader, node):
            mapping = loader.construct_mapping(node, deep=True)
            rows = mapping['rows']
            cols = mapping['cols']
            data = mapping['data']
            return np.array(data).reshape(rows, cols)

        yaml.add_constructor('!opencv-matrix', opencv_matrix_constructor)

        calib_data = yaml.load(f, Loader=yaml.FullLoader)
        
        try:
            calib_data = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(f"Error loading calibration YAML: {e}")
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
    Takes 2D DVS pixel coordinates and projects them onto the IR camera's image plane.
    
    Args:
        dvs_points (np.array): Nx2 array of DVS pixel coordinates (x, y).
        ... (calibration parameters) ...
        
    Returns:
        np.array: Nx2 array of projected IR pixel coordinates.
    """
    
    # 1. Undistort DVS points (get them into normalized coordinate system)
    # The 'None' is for the optional R matrix (rectification)
    # The output is Nx1x2
    undistorted_dvs_points = cv2.undistortPoints(
        dvs_points.reshape(-1, 1, 2), K_dvs, D_dvs, R=None, P=K_dvs
    )
    
    # 2. Setup the transformation from DVS frame to IR frame
    R_dvs_to_ir = R_ir_to_dvs.T # R is IR -> DVS, so R.T is DVS -> IR
    T_dvs_to_ir = -R_ir_to_dvs.T @ T_ir_to_dvs

    # 3. Prepare 3D points in DVS frame (assuming Z=1, as we are projecting rays)
    # The points are taken from the normalized plane (output of undistortPoints)
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

# --- 2. MAIN PLAYBACK LOGIC ---

def run():
    # Configuration
    dataset_path = "dataset"
    calib_file = "stereo_calibration_checkerboard.yaml" # Your calibration file name
    
    # --- Load Calibration ---
    K_dvs, D_dvs, K_ir, D_ir, R, T = load_calibration_data(calib_file)
    if K_dvs is None:
        print(f"Failed to load calibration data from {calib_file}.")
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
        print("No recordings found in dataset. Please create a 'dataset' folder and place files inside.")
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
        height, width = mv_iterator.get_size()
        if not is_live_camera(event_file):
            mv_iterator = LiveReplayEventsIterator(mv_iterator)

        # The frame generator height/width must match the target (IR) resolution
        # We will use the IR intrinsics matrix to determine the output size
        # K_ir[1,2]*2 and K_ir[0,2]*2 gives the approximate sensor size
        # A safer method is to wait for the first IR frame
        
        # Initialize Event Frame (size based on DVS intrinsics for the undistort step)
        event_frame_dvs_size = np.zeros((height, width, 3), dtype=np.uint8)

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal event_frame_dvs_size
            # Use decay for visualization
            alpha = 0.5
            event_frame_dvs_size = cv2.addWeighted(event_frame_dvs_size, alpha, cd_frame, 1 - alpha, 0)
        
        # We process events first, then warp the resulting image/points

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
    start_real_ts = None  # first RealSense timestamp (ms)
    leftover_events = None # Buffer for events read ahead of current frame

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
                        # Set up the event frame generator now that we know the target resolution
                        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=IR_WIDTH,
                                                                           sensor_height=IR_HEIGHT,
                                                                           fps=60,
                                                                           palette=ColorPalette.Dark)
                        # We change the callback to use the final warped image
                        event_frame = np.zeros((IR_HEIGHT, IR_WIDTH, 3), dtype=np.uint8)
                        
                        def on_cd_frame_cb_warped(ts, cd_frame):
                            nonlocal event_frame
                            # Use decay for visualization
                            alpha = 0.5
                            event_frame = cv2.addWeighted(event_frame, alpha, cd_frame, 1 - alpha, 0)
                            
                        event_frame_gen.set_output_callback(on_cd_frame_cb_warped)


                if depth_frame:
                    depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    real_ts = depth_frame.get_timestamp()  # ms
                    if start_real_ts is None:
                        start_real_ts = real_ts

                    # Convert RealSense timestamp to µs relative to start
                    t_rs = int((real_ts - start_real_ts) * 1000)

            if pipeline and real_ts is None:
                 # RealSense is running but no valid frame for timestamp alignment
                 continue
                 
            # ---------------- Event accumulation (Time Alignment) ----------------
            last_event_ts = None
            events_in_frame = 0
            
            if mv_iterator and real_ts is not None:
                accumulated_events = []
                
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
                        current_chunk = None # Consumed, get new one
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
                        break # Done for this frame

            # ---------------- Spatial Alignment & Visualization ----------------
            if accumulated_events and event_frame_gen:
                combined_events = np.concatenate(accumulated_events)
                
                # Extract event pixel coordinates
                dvs_points = np.stack((combined_events['x'], combined_events['y']), axis=1).astype(np.float32)
                
                # 1. Project DVS points to IR image plane
                # We need the events to be projected to the IR frame before being fed to the frame generator
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
                    # event_frame is already the right size (IR_HEIGHT x IR_WIDTH)
                    cv2.addWeighted(event_frame, 0.7, overlay, 0.3, 0, overlay)
                cv2.imshow("Aligned RGB + Event Overlay", overlay)

            if depth_frame is not None:
                cv2.imshow("Depth Stream", depth_color)
            
            # The 'Event Stream' window is now the warped/projected event map
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