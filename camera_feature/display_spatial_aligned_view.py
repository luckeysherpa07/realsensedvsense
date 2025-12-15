import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
import sys
from dvsense_driver.camera_manager import DvsCameraManager

# ---------------- CONFIGURATION ----------------
EVENT_BATCH_TIME = 10000     # 10ms accumulation
DEPTH_SCALE_FACTOR = 100.0   # RealSense (m) -> Calibration (cm)
VIDEO_FPS = 30

def load_calibration_yaml(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found at: {filepath}")

    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    
    def get_mat(node_name):
        node = fs.getNode(node_name)
        if node.isNone():
            raise ValueError(f"Could not find {node_name} in YAML.")
        return node.mat()

    calib = {}
    calib['K_dvs'] = get_mat("DVS_intrinsics")
    calib['D_dvs'] = get_mat("DVS_distortion")
    calib['K_ir']  = get_mat("IR_intrinsics")
    calib['D_ir']  = get_mat("IR_distortion")
    calib['R']     = get_mat("Rotation")
    calib['T']     = get_mat("Translation")
    
    fs.release()
    return calib

def apply_distortion_torch(x, y, D):
    """
    Applies distortion (k1, k2, p1, p2, k3) to normalized coordinates (x, y).
    """
    # Extract coefficients
    k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]
    
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r2*r4
    
    # Radial distortion
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    
    # Tangential distortion
    dx = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    dy = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    
    x_dist = x * radial + dx
    y_dist = y * radial + dy
    
    return x_dist, y_dist

def project_dvs_to_ir_view(dvs_img_tensor, depth_tensor, K_ir, K_dvs, D_dvs, R, T, height, width, device='cuda'):
    """
    Warps the DVS image to match the IR camera view using Depth, Intrinsics, and Distortion.
    """
    # 1. Create Grid
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(rows, cols, indexing='ij')

    # Unpack Intrinsics (IR)
    fx_ir, fy_ir = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx_ir, cy_ir = float(K_ir[0, 2]), float(K_ir[1, 2])
    
    # Unpack Intrinsics (DVS)
    fx_dvs, fy_dvs = float(K_dvs[0, 0]), float(K_dvs[1, 1])
    cx_dvs, cy_dvs = float(K_dvs[0, 2]), float(K_dvs[1, 2])

    # 2. Back-project IR pixels to 3D point cloud (P_ir)
    Z = depth_tensor
    
    # Mask invalid depth to avoid projecting garbage
    valid_mask = (Z > 0.01)
    
    X = (x - cx_ir) * Z / fx_ir
    Y = (y - cy_ir) * Z / fy_ir
    
    # Flatten: (3, N)
    P_ir = torch.stack((X, Y, Z), dim=0).reshape(3, -1)
    
    # 3. Transform to DVS Coordinate System
    R_torch = torch.from_numpy(R).to(device).float()
    T_torch = torch.from_numpy(T).to(device).float()
    
    P_dvs = torch.mm(R_torch, P_ir) + T_torch
    
    # 4. Project P_dvs onto DVS Image Plane
    X_dvs = P_dvs[0, :]
    Y_dvs = P_dvs[1, :]
    Z_dvs = P_dvs[2, :]
    
    Z_dvs[Z_dvs == 0] = 1e-6
    
    x_norm = X_dvs / Z_dvs
    y_norm = Y_dvs / Z_dvs
    
    # 5. Apply DVS Lens Distortion
    D_vals = D_dvs.flatten() if hasattr(D_dvs, 'flatten') else D_dvs
    x_dist, y_dist = apply_distortion_torch(x_norm, y_norm, D_vals)
    
    # 6. Convert to Pixels
    u_dvs = x_dist * fx_dvs + cx_dvs
    v_dvs = y_dist * fy_dvs + cy_dvs
    
    # 7. Sample from Source DVS Image
    dvs_H, dvs_W = dvs_img_tensor.shape
    
    u_grid = (u_dvs / (dvs_W - 1)) * 2 - 1
    v_grid = (v_dvs / (dvs_H - 1)) * 2 - 1
    
    grid = torch.stack((u_grid, v_grid), dim=1).reshape(1, height, width, 2)
    
    dvs_batch = dvs_img_tensor.unsqueeze(0).unsqueeze(0) 
    
    warped_dvs = torch.nn.functional.grid_sample(
        dvs_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    warped_dvs = warped_dvs.squeeze()
    warped_dvs = warped_dvs * valid_mask # Apply depth mask
    
    return warped_dvs

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))      
    parent_dir = os.path.dirname(base_dir)                     
    
    calib_path = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")
    dataset_dir = os.path.join(parent_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # We will save the IR overlay video only (as per original logic), 
    # but display both windows.
    video_path = os.path.join(dataset_dir, "overlay_result.mp4")
    print(f"Loading calibration from: {calib_path}")
    
    try:
        calib = load_calibration_yaml(calib_path)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return

    # ---------------- Initialize RealSense ----------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # Helper to colorize depth for visualization
    colorizer = rs.colorizer()
    
    print("Starting RealSense...")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    rs_scale = depth_sensor.get_depth_scale()

    # ---------------- Initialize DVS ----------------
    dvs_manager = DvsCameraManager()
    dvs_manager.update_cameras()
    if not dvs_manager.get_camera_descs():
        print("No DVS camera found.")
        pipeline.stop()
        return

    dvs_cam = dvs_manager.open_camera(dvs_manager.get_camera_descs()[0].serial)
    dvs_width, dvs_height = dvs_cam.get_width(), dvs_cam.get_height()
    dvs_cam.start()
    dvs_cam.set_batch_events_time(EVENT_BATCH_TIME)
    
    # ---------------- Video Writer ----------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (1280, 720))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing on: {device}")

    dvs_hist_gpu = torch.zeros((2, dvs_height, dvs_width), dtype=torch.float32, device=device)
    ones_gpu = None 

    try:
        while True:
            # 1. Get RealSense Data
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = frames.get_depth_frame()
            if not ir_frame or not depth_frame: continue

            ir_img = np.asanyarray(ir_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # Generate colorized depth map for visualization
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_img = np.asanyarray(depth_color_frame.get_data())

            # 2. Get DVS Data & Accumulate
            events = dvs_cam.get_next_batch()
            dvs_hist_gpu.zero_()
            
            if events['x'].size > 0:
                xs = torch.from_numpy(events['x'].astype(np.int64)).to(device)
                ys = torch.from_numpy(events['y'].astype(np.int64)).to(device)
                pols = torch.from_numpy(events['polarity'].astype(np.int64)).to(device)
                
                if ones_gpu is None or ones_gpu.shape[0] != xs.shape[0]:
                    ones_gpu = torch.ones(xs.shape[0], dtype=torch.float32, device=device)
                
                dvs_hist_gpu.index_put_((pols, ys, xs), ones_gpu, accumulate=True)

            dvs_img_raw = (dvs_hist_gpu[1] + dvs_hist_gpu[0]) 
            if dvs_img_raw.max() > 0:
                dvs_img_raw /= 5.0
                dvs_img_raw = torch.clamp(dvs_img_raw, 0, 1)
            
            # 3. Prepare Depth Tensor
            scale_conversion = rs_scale * DEPTH_SCALE_FACTOR
            depth_tensor = torch.from_numpy(depth_img).to(device).float() * scale_conversion

            # 4. Warp DVS
            warped_dvs = project_dvs_to_ir_view(
                dvs_img_raw,
                depth_tensor,
                calib['K_ir'],
                calib['K_dvs'],
                calib['D_dvs'],
                calib['R'],
                calib['T'],
                height=720,
                width=1280,
                device=device
            )

            # 5. Prepare Visualizations
            warped_np = warped_dvs.cpu().numpy()
            warped_u8 = (warped_np * 255).astype(np.uint8)
            mask_indices = warped_u8 > 0
            
            # --- Window 1: IR + Events (Magenta) ---
            ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
            ir_overlay = ir_bgr.copy()

             # FIX: Add .flatten() or .squeeze() to the cv2.add result
            if np.any(mask_indices):
                # Blue Channel
                ir_overlay[mask_indices, 0] = cv2.add(
                    ir_overlay[mask_indices, 0], 
                    warped_u8[mask_indices]
                ).flatten() 
                
                # Red Channel
                ir_overlay[mask_indices, 2] = cv2.add(
                    ir_overlay[mask_indices, 2], 
                    warped_u8[mask_indices]
                ).flatten()

            # --- Window 2: Depth + Events (White) ---
            # Depth map is already colorful (Jet usually). We add White events to make them pop.
            depth_overlay = depth_color_img.copy()
            depth_overlay[:, :, 0] = cv2.add(depth_overlay[:, :, 0], warped_u8)
            depth_overlay[:, :, 1] = cv2.add(depth_overlay[:, :, 1], warped_u8)
            depth_overlay[:, :, 2] = cv2.add(depth_overlay[:, :, 2], warped_u8)

            # Show windows
            cv2.imshow("IR & Event Alignment", ir_overlay)
            cv2.imshow("Depth & Event Alignment", depth_overlay)
            
            # Save IR result to file
            out.write(ir_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        dvs_cam.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved: {video_path}")

if __name__ == "__main__":
    run()