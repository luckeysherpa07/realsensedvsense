import os
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from metavision_core.event_io import EventsIterator

# ---------------- CONFIGURATION ----------------
DEPTH_SCALE_FACTOR = 100.0   
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DvsProjector:
    """
    Handles the 3D projection of DVS events onto the IR/Depth camera view.
    """
    def __init__(self, calib, height, width, device):
        self.device = device
        self.h, self.w = height, width
        self.calib = calib
        
        # Convert Calibration to Tensors ONCE
        self.K_ir = torch.from_numpy(calib['K_ir']).to(device).float()
        self.K_dvs = torch.from_numpy(calib['K_dvs']).to(device).float()
        self.R = torch.from_numpy(calib['R']).to(device).float()
        self.T = torch.from_numpy(calib['T']).to(device).float()
        
        # Handle Distortion Coeffs
        d_dvs = calib['D_dvs']
        self.D_dvs = torch.from_numpy(d_dvs).to(device).float().flatten()
        
        # Pre-compute Grid for IR/Depth Camera (Output View)
        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(rows, cols, indexing='ij')

    def apply_distortion(self, x, y):
        """Applies radial and tangential distortion."""
        k1, k2 = self.D_dvs[0], self.D_dvs[1]
        p1, p2 = self.D_dvs[2], self.D_dvs[3]
        k3 = self.D_dvs[4] if len(self.D_dvs) > 4 else 0.0
        
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4
        
        radial = 1.0 + k1*r2 + k2*r4 + k3*r6
        dx = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
        dy = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
        
        return x * radial + dx, y * radial + dy

    def project(self, dvs_tensor, depth_tensor, off_x=0.0, off_y=0.0):
        """
        dvs_tensor: (C, H, W) Event Image
        depth_tensor: (H, W) Depth Map
        off_x, off_y: Manual pixel shift corrections
        """
        fx_ir, fy_ir = self.K_ir[0, 0], self.K_ir[1, 1]
        cx_ir, cy_ir = self.K_ir[0, 2], self.K_ir[1, 2]
        
        fx_dvs, fy_dvs = self.K_dvs[0, 0], self.K_dvs[1, 1]
        cx_dvs, cy_dvs = self.K_dvs[0, 2], self.K_dvs[1, 2]

        # 1. Back-project IR/Depth pixels to 3D
        Z = depth_tensor
        valid_mask = (Z > 0.01) # Filter invalid depth

        X = (self.grid_x - cx_ir) * Z / fx_ir
        Y = (self.grid_y - cy_ir) * Z / fy_ir
        
        P_ir = torch.stack((X, Y, Z), dim=0).reshape(3, -1)
        
        # 2. Transform to DVS Coords
        P_dvs = torch.mm(self.R, P_ir) + self.T
        
        # 3. Project to DVS Image Plane
        X_dvs, Y_dvs, Z_dvs = P_dvs[0], P_dvs[1], P_dvs[2]
        
        Z_dvs[Z_dvs == 0] = 1e-6
        x_norm = X_dvs / Z_dvs
        y_norm = Y_dvs / Z_dvs
        
        # 4. Apply Distortion & Intrinsics
        x_dist, y_dist = self.apply_distortion(x_norm, y_norm)
        
        u_dvs = x_dist * fx_dvs + cx_dvs
        v_dvs = y_dist * fy_dvs + cy_dvs
        
        # --- APPLY MANUAL OFFSET ---
        u_dvs += off_x
        v_dvs += off_y
        # ---------------------------

        # 5. Grid Sample
        dvs_H, dvs_W = dvs_tensor.shape[-2:]
        
        u_grid = (u_dvs / (dvs_W - 1)) * 2 - 1
        v_grid = (v_dvs / (dvs_H - 1)) * 2 - 1
        
        grid = torch.stack((u_grid, v_grid), dim=1).reshape(1, self.h, self.w, 2)
        
        if dvs_tensor.ndim == 2:
            dvs_batch = dvs_tensor.unsqueeze(0).unsqueeze(0)
        else:
            dvs_batch = dvs_tensor.unsqueeze(0)

        warped = torch.nn.functional.grid_sample(
            dvs_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return warped.squeeze() * valid_mask

def load_calibration_yaml(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Calibration file not found at: {filepath}")
        return None

    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    calib = {}
    keys = {
        'DVS_intrinsics': 'K_dvs', 'DVS_distortion': 'D_dvs', 
        'IR_intrinsics': 'K_ir', 'IR_distortion': 'D_ir', 
        'Rotation': 'R', 'Translation': 'T'
    }
    
    for yaml_key, dict_key in keys.items():
        node = fs.getNode(yaml_key)
        if node.isNone():
            print(f"Error: Missing key '{yaml_key}'")
            return None
        calib[dict_key] = node.mat()
    fs.release()
    return calib

def run():
    # ---------------- PATH SETUP ----------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(parent_dir, "dataset")
    calib_file = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at: {dataset_path}")
        return

    # ---------------- FILE SELECTION ----------------
    files = os.listdir(dataset_path)
    prefixes = sorted(list(set([f.replace("_event.raw", "") for f in files if f.endswith("_event.raw")])))
    if not prefixes:
        print("No recordings found.")
        return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1):
        print(f"{idx}. {prefix}")

    choice = input(f"\nSelect a recording (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1: return
    selected_prefix = prefixes[int(choice) - 1]

    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # ---------------- LOAD RESOURCES ----------------
    print(f"Processing on: {DEVICE}")
    calib_data = load_calibration_yaml(calib_file)
    
    delta_us = 0
    if os.path.exists(delta_file):
        try:
            with open(delta_file, "r") as f:
                delta_us = int(float(f.read().strip()) * 1e6)
            print(f"Time offset loaded: {delta_us / 1e6:.4f} s")
        except: pass

    # ---------------- SETUP EVENT READER ----------------
    def reset_event_iterator():
        it = EventsIterator(input_path=event_file, delta_t=10000) # 10ms chunks
        h, w = it.get_size()
        return it, iter(it), h, w

    mv_iterator, ev_iter, ev_h, ev_w = reset_event_iterator()
    dvs_hist_gpu = torch.zeros((2, ev_h, ev_w), dtype=torch.float32, device=DEVICE)
    projector = None

    # ---------------- SETUP REALSENSE ----------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=True)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(True)
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    
    # State variables
    rs_start_ts_ms = None
    prev_ts_ms = -1
    leftover_events = None
    ev_start_ts_us = None

    # ---------------- MANUAL OFFSET VARIABLES ----------------
    offset_x = 0.0
    offset_y = 0.0

    print("\n---------------- CONTROLS ----------------")
    print(" [W/S]: Shift Vertical")
    print(" [A/D]: Shift Horizontal")
    print(" [ESC]: Quit")
    print("------------------------------------------\n")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break 

            depth_frame = frames.get_depth_frame()
            if not depth_frame: continue

            # Timing & Loop Detection
            curr_ms = frames.get_timestamp()
            if prev_ts_ms > 0 and curr_ms < prev_ts_ms:
                mv_iterator, ev_iter, _, _ = reset_event_iterator()
                leftover_events = None
                ev_start_ts_us = None
            
            prev_ts_ms = curr_ms
            if rs_start_ts_ms is None: rs_start_ts_ms = curr_ms
            
            elapsed_us = int((curr_ms - rs_start_ts_ms) * 1000)

            # Images
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # Init Projector
            if projector is None and calib_data:
                projector = DvsProjector(calib_data, depth_img.shape[0], depth_img.shape[1], DEVICE)

            # ---------------- EVENT PROCESSING ----------------
            dvs_hist_gpu.zero_()
            
            if ev_iter:
                while ev_start_ts_us is None:
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]
                        break
                    try:
                        leftover_events = next(ev_iter)
                    except StopIteration:
                        ev_iter = None; break
                
                if ev_iter and ev_start_ts_us is not None:
                    target_ev = ev_start_ts_us + elapsed_us - delta_us
                    
                    if target_ev > ev_start_ts_us:
                        accumulated = []
                        if leftover_events is not None and len(leftover_events) > 0:
                            accumulated.append(leftover_events)
                            leftover_events = None
                        
                        cur_t = accumulated[-1]['t'][-1] if accumulated else ev_start_ts_us
                        
                        while cur_t < target_ev:
                            try:
                                nb = next(ev_iter)
                                if len(nb) == 0: continue
                                accumulated.append(nb)
                                cur_t = nb['t'][-1]
                            except StopIteration:
                                ev_iter = None; break
                        
                        if accumulated:
                            all_evs = np.concatenate(accumulated)
                            split_idx = np.searchsorted(all_evs['t'], target_ev, side='right')
                            events_now = all_evs[:split_idx]
                            leftover_events = all_evs[split_idx:]
                            
                            if len(events_now) > 0:
                                xs = torch.from_numpy(events_now['x'].astype(np.int64)).to(DEVICE)
                                ys = torch.from_numpy(events_now['y'].astype(np.int64)).to(DEVICE)
                                ps = torch.from_numpy(events_now['p'].astype(np.int64)).to(DEVICE)
                                
                                mask_ev = (xs < ev_w) & (ys < ev_h)
                                if mask_ev.any():
                                    dvs_hist_gpu.index_put_(
                                        (ps[mask_ev], ys[mask_ev], xs[mask_ev]), 
                                        torch.tensor(1.0, device=DEVICE), accumulate=True
                                    )

            # ---------------- WARPING & VISUALIZATION ----------------
            dvs_raw_clr = torch.stack((dvs_hist_gpu[0], torch.zeros_like(dvs_hist_gpu[0]), dvs_hist_gpu[1]), dim=0)
            
            val_max = dvs_raw_clr.max()
            if val_max > 0:
                dvs_raw_clr = torch.clamp(dvs_raw_clr / 3.0, 0, 1)

            # --- VISUALIZATION UPDATE ---
            # Used convertScaleAbs with alpha=0.08 to match the requested look
            # This provides better contrast in the near-range
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.08), 
                cv2.COLORMAP_JET
            )

            if projector:
                depth_tensor = torch.from_numpy(depth_img).to(DEVICE).float() * (depth_scale * DEPTH_SCALE_FACTOR)
                
                # Project events
                warped_dvs = projector.project(dvs_raw_clr, depth_tensor, off_x=offset_x, off_y=offset_y)
                warped_u8 = (warped_dvs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Overlay
                mask = np.any(warped_u8 > 10, axis=2)
                if np.any(mask):
                    depth_colormap[mask] = cv2.addWeighted(depth_colormap[mask], 0.7, warped_u8[mask], 1.0, 0)

            # Draw Offset Text
            cv2.putText(depth_colormap, f"Offset X: {offset_x:.1f} | Y: {offset_y:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Depth + DVS Projection", depth_colormap)

            # ---------------- KEYBOARD CONTROL ----------------
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key == ord('w'): # Move UP
                offset_y -= 1.0
            elif key == ord('s'): # Move DOWN
                offset_y += 1.0
            elif key == ord('a'): # Move LEFT
                offset_x -= 1.0
            elif key == ord('d'): # Move RIGHT
                offset_x += 1.0

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()