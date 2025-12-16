import os
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from metavision_core.event_io import EventsIterator

# ---------------- CONFIGURATION ----------------
# Scaling factor: RealSense uses Meters. 
# If calibration (checkerboard) was in Centimeters, use 100.0.
# If calibration was in Meters, use 1.0.
DEPTH_SCALE_FACTOR = 100.0   

def load_calibration_yaml(filepath):
    """Loads calibration matrices from OpenCV YAML format."""
    if not os.path.exists(filepath):
        print(f"Error: Calibration file not found at: {filepath}")
        return None

    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    
    def get_mat(node_name):
        node = fs.getNode(node_name)
        if node.isNone():
            return None
        return node.mat()

    calib = {}
    calib['K_dvs'] = get_mat("DVS_intrinsics")
    calib['D_dvs'] = get_mat("DVS_distortion")
    calib['K_ir']  = get_mat("IR_intrinsics")
    calib['D_ir']  = get_mat("IR_distortion")
    calib['R']     = get_mat("Rotation")
    calib['T']     = get_mat("Translation")
    
    fs.release()
    
    # Check completeness
    for k, v in calib.items():
        if v is None:
            print(f"Error: Missing key '{k}' in calibration file.")
            return None
            
    return calib

def apply_distortion_torch(x, y, D):
    """Applies distortion to normalized coordinates."""
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4] if len(D) > 4 else 0.0
    
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r2*r4
    
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    dx = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    dy = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    
    x_dist = x * radial + dx
    y_dist = y * radial + dy
    return x_dist, y_dist

def project_dvs_to_ir_view(dvs_img_tensor, depth_tensor, calib, height, width, device='cuda'):
    """
    Warps the DVS image to match the IR camera view using Depth and Calibration.
    """
    K_ir, K_dvs = calib['K_ir'], calib['K_dvs']
    D_dvs = calib['D_dvs']
    R, T = calib['R'], calib['T']

    # 1. Create Grid
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(rows, cols, indexing='ij')

    fx_ir, fy_ir = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx_ir, cy_ir = float(K_ir[0, 2]), float(K_ir[1, 2])
    
    fx_dvs, fy_dvs = float(K_dvs[0, 0]), float(K_dvs[1, 1])
    cx_dvs, cy_dvs = float(K_dvs[0, 2]), float(K_dvs[1, 2])

    # 2. Back-project IR pixels to 3D
    Z = depth_tensor
    valid_mask = (Z > 0.01) 
    
    X = (x - cx_ir) * Z / fx_ir
    Y = (y - cy_ir) * Z / fy_ir
    
    P_ir = torch.stack((X, Y, Z), dim=0).reshape(3, -1)
    
    # 3. Transform to DVS Coords
    R_torch = torch.from_numpy(R).to(device).float()
    T_torch = torch.from_numpy(T).to(device).float()
    
    P_dvs = torch.mm(R_torch, P_ir) + T_torch
    
    # 4. Project onto DVS Image Plane
    X_dvs = P_dvs[0, :]
    Y_dvs = P_dvs[1, :]
    Z_dvs = P_dvs[2, :]
    
    Z_dvs[Z_dvs == 0] = 1e-6
    
    x_norm = X_dvs / Z_dvs
    y_norm = Y_dvs / Z_dvs
    
    # 5. Apply Distortion
    D_vals = D_dvs.flatten() if hasattr(D_dvs, 'flatten') else D_dvs
    x_dist, y_dist = apply_distortion_torch(x_norm, y_norm, D_vals)
    
    # 6. Convert to Pixels
    u_dvs = x_dist * fx_dvs + cx_dvs
    v_dvs = y_dist * fy_dvs + cy_dvs
    
    # 7. Sample
    dvs_H, dvs_W = dvs_img_tensor.shape
    
    u_grid = (u_dvs / (dvs_W - 1)) * 2 - 1
    v_grid = (v_dvs / (dvs_H - 1)) * 2 - 1
    
    grid = torch.stack((u_grid, v_grid), dim=1).reshape(1, height, width, 2)
    dvs_batch = dvs_img_tensor.unsqueeze(0).unsqueeze(0) 
    
    warped_dvs = torch.nn.functional.grid_sample(
        dvs_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    return warped_dvs.squeeze() * valid_mask

def run():
    # ---------------- PATH SETUP ----------------
    # Script location: .../camera_feature/script.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parent directory: .../
    parent_dir = os.path.dirname(script_dir)
    
    # Dataset and Calibration paths
    dataset_path = os.path.join(parent_dir, "dataset")
    calib_file = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at: {dataset_path}")
        return
    
    # ---------------- DEVICE SETUP ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing using: {device}")

    # ---------------- FILE SELECTION ----------------
    files = os.listdir(dataset_path)
    prefixes = sorted(list(set([f.replace("_event.raw", "") for f in files if f.endswith("_event.raw")])))

    if not prefixes:
        print(f"No recordings found in {dataset_path}")
        return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1):
        print(f"{idx}. {prefix}")

    choice = input(f"\nSelect a recording (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(prefixes):
        print("Invalid choice.")
        return
    selected_prefix = prefixes[int(choice) - 1]

    # Recording Paths
    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")
    
    # ---------------- LOAD RESOURCES ----------------
    print(f"Loading calibration from: {calib_file}")
    calib_data = load_calibration_yaml(calib_file)
    
    if calib_data is None:
        print("Warning: Calibration failed to load. Warping will be disabled.")
    
    delta_us = 0
    if os.path.exists(delta_file):
        try:
            with open(delta_file, "r") as f:
                val = float(f.read().strip())
                delta_us = int(val * 1e6)
            print(f"Time offset loaded: {delta_us / 1e6} s")
        except: pass

    # ---------------- SETUP EVENT READER ----------------
    if not os.path.exists(event_file): return
    mv_iterator = EventsIterator(input_path=event_file, delta_t=1000)
    ev_height, ev_width = mv_iterator.get_size()
    ev_iter = iter(mv_iterator)
    
    dvs_hist_gpu = torch.zeros((2, ev_height, ev_width), dtype=torch.float32, device=device)
    ones_gpu = None 
    leftover_events = None 
    ev_start_ts_us = None

    # ---------------- SETUP REALSENSE ----------------
    if not os.path.exists(bag_file): return
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(True) 
    
    colorizer = rs.colorizer()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("\nStarting playback...")
    rs_start_ts_ms = None

    try:
        while True:
            # 1. RealSense Frame
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print("End of bag file.")
                break 

            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1) or frames.get_infrared_frame(0)
            if not depth_frame or not ir_frame: continue

            # Timing
            curr_ms = frames.get_timestamp()
            if rs_start_ts_ms is None: rs_start_ts_ms = curr_ms
            elapsed_us = int((curr_ms - rs_start_ts_ms) * 1000)

            # Images
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            ir_raw = np.asanyarray(ir_frame.get_data())
            if ir_raw.dtype == np.uint16:
                ir_8bit = cv2.normalize(ir_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                ir_8bit = ir_raw.astype(np.uint8)
            ir_bgr = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

            rs_h, rs_w = depth_img.shape

            # 2. Event Slicing (FIXED LOGIC)
            events_to_process = None
            if ev_iter:
                # --- Step A: Ensure we have a valid Start Timestamp ---
                while ev_start_ts_us is None:
                    # Check if we have pending events from previous read
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]
                        break
                    
                    # If not, try to fetch the first batch
                    try:
                        leftover_events = next(ev_iter)
                    except StopIteration:
                        print("Event file is empty or finished.")
                        ev_iter = None
                        break
                
                # If we still don't have a timestamp, we can't process events
                if ev_iter is None or ev_start_ts_us is None:
                    # Skip event processing this frame, just show video
                    pass 
                else:
                    # --- Step B: Calculate Target and Fetch ---
                    target_ev = ev_start_ts_us + elapsed_us - delta_us
                    
                    accumulated = []
                    if leftover_events is not None:
                        accumulated.append(leftover_events)
                        leftover_events = None
                    
                    # Find max time in current buffer
                    cur_t = accumulated[-1]['t'][-1] if accumulated else 0
                    
                    # Accumulate until target
                    while cur_t < target_ev:
                        try:
                            nb = next(ev_iter)
                            if len(nb) == 0: continue
                            accumulated.append(nb)
                            cur_t = nb['t'][-1]
                        except StopIteration:
                            break
                    
                    if accumulated:
                        all_evs = np.concatenate(accumulated)
                        split = np.searchsorted(all_evs['t'], target_ev, side='right')
                        events_to_process = all_evs[:split]
                        leftover_events = all_evs[split:]

            # 3. GPU Accumulation
            dvs_hist_gpu.zero_()
            if events_to_process is not None and len(events_to_process) > 0:
                xs = torch.from_numpy(events_to_process['x'].astype(np.int64)).to(device)
                ys = torch.from_numpy(events_to_process['y'].astype(np.int64)).to(device)
                pols = torch.from_numpy(events_to_process['p'].astype(np.int64)).to(device)
                
                if ones_gpu is None or ones_gpu.shape[0] != xs.shape[0]:
                    ones_gpu = torch.ones(xs.shape[0], dtype=torch.float32, device=device)
                
                dvs_hist_gpu.index_put_((pols, ys, xs), ones_gpu, accumulate=True)

            dvs_raw = (dvs_hist_gpu[1] + dvs_hist_gpu[0])
            if dvs_raw.max() > 0:
                dvs_raw /= 3.0
                dvs_raw = torch.clamp(dvs_raw, 0, 1)

            # 4. Warping
            warped_u8 = None
            if calib_data:
                depth_tensor = torch.from_numpy(depth_img).to(device).float() * (depth_scale * DEPTH_SCALE_FACTOR)
                warped_dvs = project_dvs_to_ir_view(dvs_raw, depth_tensor, calib_data, rs_h, rs_w, device)
                warped_u8 = (warped_dvs.cpu().numpy() * 255).astype(np.uint8)
            else:
                warped_u8 = cv2.resize((dvs_raw.cpu().numpy() * 255).astype(np.uint8), (rs_w, rs_h))

            # 5. Display
            mask = warped_u8 > 10
            
            # IR Overlay (Magenta)
            ir_disp = ir_bgr.copy()
            if np.any(mask):
                ir_disp[mask, 0] = cv2.add(ir_disp[mask, 0], warped_u8[mask]).flatten()
                ir_disp[mask, 2] = cv2.add(ir_disp[mask, 2], warped_u8[mask]).flatten()
            
            # Depth Overlay (White)
            depth_disp = depth_color.copy()
            if np.any(mask):
                depth_disp[mask] = [255, 255, 255]

            cv2.imshow("IR + Warped Events", ir_disp)
            cv2.imshow("Depth + Warped Events", depth_disp)

            if cv2.waitKey(1) == 27: break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()