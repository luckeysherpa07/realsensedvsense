import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
import sys
from dvsense_driver.camera_manager import DvsCameraManager

# ---------------- CONFIGURATION ----------------
EVENT_BATCH_TIME = 10000     # 10ms accumulation
DEPTH_SCALE_FACTOR = 100.0   # RealSense (m) -> Calibration (cm). 1m = 100cm.
VIDEO_FPS = 30

def load_calibration_yaml(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found at: {filepath}")

    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    
    def get_mat(node_name):
        mat = fs.getNode(node_name).mat()
        if mat is None:
            raise ValueError(f"Could not find {node_name} in YAML.")
        return mat

    calib = {}
    calib['K_dvs'] = get_mat("DVS_intrinsics")
    calib['D_dvs'] = get_mat("DVS_distortion")
    calib['K_ir']  = get_mat("IR_intrinsics")
    calib['D_ir']  = get_mat("IR_distortion")
    calib['R']     = get_mat("Rotation")
    calib['T']     = get_mat("Translation")
    
    fs.release()
    return calib

def project_dvs_to_ir_view(dvs_img_tensor, depth_tensor, K_ir, K_dvs, R, T, height, width, device='cuda'):
    """
    Warps the DVS image to match the IR camera view using Depth.
    Performed on GPU for speed.
    """
    # 1. Create Grid Manually to ensure (Height, Width) dimensions
    # height should be 720, width should be 1280
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    
    # y: varies down rows (720, 1280)
    y = rows.view(-1, 1).repeat(1, width)
    # x: varies across columns (720, 1280)
    x = cols.view(1, -1).repeat(height, 1)

    # Unpack Intrinsics (IR)
    fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
    cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
    
    # Unpack Intrinsics (DVS)
    fx_dvs, fy_dvs = K_dvs[0, 0], K_dvs[1, 1]
    cx_dvs, cy_dvs = K_dvs[0, 2], K_dvs[1, 2]

    # 2. Back-project IR pixels to 3D point cloud (P_ir)
    Z = depth_tensor # Expecting (720, 1280)
    
    # x, y, Z must all be (720, 1280)
    X = (x - cx_ir) * Z / fx_ir
    Y = (y - cy_ir) * Z / fy_ir
    
    # Shape: (3, H*W)
    P_ir = torch.stack((X, Y, Z), dim=0).reshape(3, -1)
    
    # 3. Transform to DVS Coordinate System: P_dvs = R * P_ir + T
    R_torch = torch.from_numpy(R).to(device).float()
    T_torch = torch.from_numpy(T).to(device).float()
    
    P_dvs = torch.mm(R_torch, P_ir) + T_torch
    
    # 4. Project P_dvs onto DVS Image Plane
    X_dvs = P_dvs[0, :]
    Y_dvs = P_dvs[1, :]
    Z_dvs = P_dvs[2, :]
    
    # Avoid division by zero
    Z_dvs[Z_dvs == 0] = 1e-6
    
    u_dvs = (X_dvs / Z_dvs) * fx_dvs + cx_dvs
    v_dvs = (Y_dvs / Z_dvs) * fy_dvs + cy_dvs
    
    # 5. Sample from Source DVS Image
    dvs_H, dvs_W = dvs_img_tensor.shape
    
    u_norm = (u_dvs / (dvs_W - 1)) * 2 - 1
    v_norm = (v_dvs / (dvs_H - 1)) * 2 - 1
    
    grid = torch.stack((u_norm, v_norm), dim=1).reshape(1, height, width, 2)
    
    dvs_batch = dvs_img_tensor.unsqueeze(0).unsqueeze(0) 
    
    warped_dvs = torch.nn.functional.grid_sample(
        dvs_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    return warped_dvs.squeeze()

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))      
    parent_dir = os.path.dirname(base_dir)                     
    
    calib_path = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")
    dataset_dir = os.path.join(parent_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
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

    try:
        while True:
            # 1. Get Data
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = frames.get_depth_frame()
            if not ir_frame or not depth_frame: continue

            ir_img = np.asanyarray(ir_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # Check dimensions just in case
            h, w = ir_img.shape
            if h != 720 or w != 1280:
                print(f"Unexpected resolution: {w}x{h}")
                continue

            events = dvs_cam.get_next_batch()
            if events['x'].size == 0: continue
                
            xs = torch.from_numpy(events['x'].astype(np.int32)).long()
            ys = torch.from_numpy(events['y'].astype(np.int32)).long()
            pols = torch.from_numpy(events['polarity'].astype(np.int32)).long()
            
            dvs_hist = torch.zeros((2, dvs_height, dvs_width), dtype=torch.float32)
            dvs_hist.index_put_((pols, ys, xs), torch.ones_like(xs, dtype=torch.float32), accumulate=True)
            dvs_img_raw = (dvs_hist[1] + dvs_hist[0]) 
            
            if dvs_img_raw.max() > 0:
                dvs_img_raw /= 5.0
                dvs_img_raw = torch.clamp(dvs_img_raw, 0, 1)

            dvs_gpu = dvs_img_raw.to(device)
            
            # 2. Prepare Depth
            scale_conversion = rs_scale * DEPTH_SCALE_FACTOR
            depth_tensor = torch.from_numpy(depth_img).to(device).float() * scale_conversion

            # 3. Warp
            # CORRECT CALL: height=720, width=1280 to match IR/Depth image
            warped_dvs = project_dvs_to_ir_view(
                dvs_gpu,
                depth_tensor,
                calib['K_ir'],
                calib['K_dvs'],
                calib['R'],
                calib['T'],
                height=720, # Matches ir_img.shape[0]
                width=1280, # Matches ir_img.shape[1]
                device=device
            )

            # 4. Visualize
            warped_np = warped_dvs.cpu().numpy()
            warped_u8 = (warped_np * 255).astype(np.uint8)
            
            ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
            
            overlay = ir_bgr.copy()
            # Add Events (Magenta)
            overlay[:, :, 0] = cv2.add(overlay[:, :, 0], warped_u8) # Blue
            overlay[:, :, 2] = cv2.add(overlay[:, :, 2], warped_u8) # Red
            
            cv2.imshow("Depth-based Extrinsic Alignment", overlay)
            out.write(overlay)

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