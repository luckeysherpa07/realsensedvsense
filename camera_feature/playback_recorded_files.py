import os
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from metavision_core.event_io import EventsIterator

# ---------------- CONFIGURATION ----------------
DEPTH_SCALE_FACTOR = 100.0   
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOV_CROP = 0.80  # Keeps inner 80% to remove fish-eye edges

# Visualization Tile Size (Height)
TILE_HEIGHT = 300 

class DvsProjector:
    def __init__(self, calib, height, width, device):
        self.device = device
        self.h, self.w = height, width
        self.calib = calib
        
        # Calibration Tensors
        self.K_ir = torch.from_numpy(calib['K_ir']).to(device).float()
        self.K_dvs = torch.from_numpy(calib['K_dvs']).to(device).float()
        self.R = torch.from_numpy(calib['R']).to(device).float()
        self.T = torch.from_numpy(calib['T']).to(device).float()
        
        # Distortion
        self.D_dvs = torch.from_numpy(calib['D_dvs']).to(device).float().flatten()
        
        # Grid
        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(rows, cols, indexing='ij')

    def apply_distortion(self, x, y):
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

    def project(self, dvs_tensor, depth_tensor, off_x=0.0, off_y=0.0, crop_factor=0.8):
        fx_ir, fy_ir = self.K_ir[0, 0], self.K_ir[1, 1]
        cx_ir, cy_ir = self.K_ir[0, 2], self.K_ir[1, 2]
        fx_dvs, fy_dvs = self.K_dvs[0, 0], self.K_dvs[1, 1]
        cx_dvs, cy_dvs = self.K_dvs[0, 2], self.K_dvs[1, 2]

        # 1. Back-project
        Z = depth_tensor
        valid_depth_mask = (Z > 0.01)
        X = (self.grid_x - cx_ir) * Z / fx_ir
        Y = (self.grid_y - cy_ir) * Z / fy_ir
        
        # 2. Transform (Flattened)
        P_ir = torch.stack((X, Y, Z), dim=0).reshape(3, -1)
        P_dvs = torch.mm(self.R, P_ir) + self.T
        X_dvs, Y_dvs, Z_dvs = P_dvs[0], P_dvs[1], P_dvs[2]
        
        # 3. Filter Back-projections (Ghosts)
        mask_dvs_front_1d = (Z_dvs > 0.05)
        Z_dvs[~mask_dvs_front_1d] = 1.0 
        
        x_norm = X_dvs / Z_dvs
        y_norm = Y_dvs / Z_dvs
        
        # 4. Distort & Project
        x_dist, y_dist = self.apply_distortion(x_norm, y_norm)
        u_dvs = x_dist * fx_dvs + cx_dvs
        v_dvs = y_dist * fy_dvs + cy_dvs
        u_dvs += off_x
        v_dvs += off_y

        # 5. Grid Sample
        dvs_H, dvs_W = dvs_tensor.shape[-2:]
        u_grid = (u_dvs / (dvs_W - 1)) * 2 - 1
        v_grid = (v_dvs / (dvs_H - 1)) * 2 - 1
        
        grid = torch.stack((u_grid, v_grid), dim=1).reshape(1, self.h, self.w, 2)
        
        # 6. Masks
        in_bounds = (grid[..., 0].abs() <= crop_factor) & (grid[..., 1].abs() <= crop_factor)
        mask_dvs_front_2d = mask_dvs_front_1d.view(self.h, self.w)
        final_mask = valid_depth_mask & mask_dvs_front_2d & in_bounds.squeeze()

        # 7. Warp
        if dvs_tensor.ndim == 2: dvs_batch = dvs_tensor[None, None, ...]
        else: dvs_batch = dvs_tensor[None, ...]

        warped = torch.nn.functional.grid_sample(
            dvs_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        return warped.squeeze(), final_mask

def load_calibration_yaml(filepath):
    if not os.path.exists(filepath): return None
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    calib = {}
    keys = {'DVS_intrinsics': 'K_dvs', 'DVS_distortion': 'D_dvs', 'IR_intrinsics': 'K_ir', 'IR_distortion': 'D_ir', 'Rotation': 'R', 'Translation': 'T'}
    for yk, dk in keys.items():
        node = fs.getNode(yk)
        if node.isNone(): return None
        calib[dk] = node.mat()
    fs.release()
    return calib

def resize_maintain_aspect(image, target_height):
    h, w = image.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    return cv2.resize(image, (target_width, target_height))

def add_label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

def run():
    # --- PATHS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(parent_dir, "dataset")
    calib_file = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")

    if not os.path.exists(dataset_path): return

    files = os.listdir(dataset_path)
    prefixes = sorted(list(set([f.replace("_event.raw", "") for f in files if f.endswith("_event.raw")])))
    if not prefixes: return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1): print(f"{idx}. {prefix}")
    choice = input(f"\nSelect (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1: return
    selected_prefix = prefixes[int(choice) - 1]

    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # --- LOAD ---
    calib_data = load_calibration_yaml(calib_file)
    delta_us = 0
    if os.path.exists(delta_file):
        try:
            with open(delta_file, "r") as f: delta_us = int(float(f.read().strip()) * 1e6)
        except: pass

    # --- RESOURCES ---
    def reset_event_iterator():
        it = EventsIterator(input_path=event_file, delta_t=10000)
        h, w = it.get_size()
        return it, iter(it), h, w

    mv_iterator, ev_iter, ev_h, ev_w = reset_event_iterator()
    dvs_hist_gpu = torch.zeros((2, ev_h, ev_w), dtype=torch.float32, device=DEVICE)
    projector = None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=True)
    profile = pipeline.start(config)
    
    # Get Scales & Devices
    dev = profile.get_device()
    playback = dev.as_playback()
    playback.set_real_time(True)
    depth_scale = dev.first_depth_sensor().get_depth_scale()

    rs_start_ts_ms = None
    prev_ts_ms = -1
    leftover_events = None
    ev_start_ts_us = None
    offset_x, offset_y = 0.0, 0.0

    print("\n[Controls] W/S/A/D: Shift Overlay | ESC: Quit")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError: break

            # Get All Frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame    = frames.get_infrared_frame(1) # Index 1 is usually Left IR

            if not depth_frame: continue
            
            # --- SYNC TIME ---
            curr_ms = frames.get_timestamp()
            if prev_ts_ms > 0 and curr_ms < prev_ts_ms:
                mv_iterator, ev_iter, _, _ = reset_event_iterator()
                leftover_events = None
                ev_start_ts_us = None
            prev_ts_ms = curr_ms
            if rs_start_ts_ms is None: rs_start_ts_ms = curr_ms
            elapsed_us = int((curr_ms - rs_start_ts_ms) * 1000)

            # --- PREPARE IMAGES ---
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # RGB (1) - FIXED COLOR ISSUE
            if color_frame:
                rgb_img = np.asanyarray(color_frame.get_data())
                # Removed cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) because stream is likely already BGR
            else:
                rgb_img = np.zeros((300, 400, 3), dtype=np.uint8)

            # IR (2)
            if ir_frame:
                ir_img_gray = np.asanyarray(ir_frame.get_data())
                ir_img_clr = cv2.cvtColor(ir_img_gray, cv2.COLOR_GRAY2BGR)
            else:
                ir_img_gray = np.zeros_like(depth_img, dtype=np.uint8)
                ir_img_clr = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

            # --- INIT PROJECTOR ---
            if projector is None and calib_data:
                projector = DvsProjector(calib_data, depth_img.shape[0], depth_img.shape[1], DEVICE)

            # --- EVENT PROCESSING ---
            dvs_hist_gpu.zero_()
            if ev_iter:
                while ev_start_ts_us is None:
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]; break
                    try: leftover_events = next(ev_iter)
                    except StopIteration: ev_iter = None; break
                
                if ev_iter and ev_start_ts_us is not None:
                    target_ev = ev_start_ts_us + elapsed_us - delta_us
                    if target_ev > ev_start_ts_us:
                        accumulated = []
                        if leftover_events is not None:
                            accumulated.append(leftover_events); leftover_events = None
                        cur_t = accumulated[-1]['t'][-1] if accumulated else ev_start_ts_us
                        while cur_t < target_ev:
                            try:
                                nb = next(ev_iter)
                                if len(nb)==0: continue
                                accumulated.append(nb); cur_t = nb['t'][-1]
                            except StopIteration: ev_iter = None; break
                        if accumulated:
                            all_evs = np.concatenate(accumulated)
                            idx = np.searchsorted(all_evs['t'], target_ev, side='right')
                            events_now, leftover_events = all_evs[:idx], all_evs[idx:]
                            if len(events_now) > 0:
                                xs = torch.from_numpy(events_now['x'].astype(np.int64)).to(DEVICE)
                                ys = torch.from_numpy(events_now['y'].astype(np.int64)).to(DEVICE)
                                ps = torch.from_numpy(events_now['p'].astype(np.int64)).to(DEVICE)
                                mask_ev = (xs < ev_w) & (ys < ev_h)
                                if mask_ev.any():
                                    dvs_hist_gpu.index_put_((ps[mask_ev], ys[mask_ev], xs[mask_ev]), torch.tensor(1.0, device=DEVICE), accumulate=True)

            # --- PREPARE VIEWS ---
            
            # 3. Event (Raw)
            dvs_raw_clr = torch.stack((dvs_hist_gpu[0], torch.zeros_like(dvs_hist_gpu[0]), dvs_hist_gpu[1]), dim=0)
            if dvs_raw_clr.max() > 0: dvs_raw_clr = torch.clamp(dvs_raw_clr / 3.0, 0, 1)
            event_img_raw = (dvs_raw_clr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # 4. Depth (Colormap)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.08), cv2.COLORMAP_JET)

            # 5 & 6 Overlays (Init with backgrounds)
            view_ir_ov = ir_img_clr.copy()
            view_depth_ov = depth_colormap.copy()

            if projector:
                depth_tensor = torch.from_numpy(depth_img).to(DEVICE).float() * (depth_scale * DEPTH_SCALE_FACTOR)
                
                warped_dvs, fov_mask = projector.project(
                    dvs_raw_clr, depth_tensor, off_x=offset_x, off_y=offset_y, crop_factor=FOV_CROP
                )
                
                mask_cpu = fov_mask.cpu().numpy().astype(bool)
                warped_u8 = (warped_dvs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Black out background outside FOV
                view_ir_ov[~mask_cpu] = 0
                view_depth_ov[~mask_cpu] = 0
                warped_u8[~mask_cpu] = 0

                # Apply Overlay
                mask_ev = np.any(warped_u8 > 10, axis=2)
                if np.any(mask_ev):
                    view_ir_ov[mask_ev] = cv2.addWeighted(view_ir_ov[mask_ev], 0.6, warped_u8[mask_ev], 1.0, 0)
                    view_depth_ov[mask_ev] = cv2.addWeighted(view_depth_ov[mask_ev], 0.6, warped_u8[mask_ev], 1.0, 0)

            # --- RESIZE AND STACK ---
            # Resize all to same height
            v1 = resize_maintain_aspect(rgb_img, TILE_HEIGHT)
            v2 = resize_maintain_aspect(ir_img_clr, TILE_HEIGHT)
            v3 = resize_maintain_aspect(event_img_raw, TILE_HEIGHT)
            
            v4 = resize_maintain_aspect(depth_colormap, TILE_HEIGHT)
            v5 = resize_maintain_aspect(view_ir_ov, TILE_HEIGHT)
            v6 = resize_maintain_aspect(view_depth_ov, TILE_HEIGHT)

            # Force uniform widths for stacking (based on IR width)
            target_w = v2.shape[1] 
            v1 = cv2.resize(v1, (target_w, TILE_HEIGHT))
            v3 = cv2.resize(v3, (target_w, TILE_HEIGHT))
            v4 = cv2.resize(v4, (target_w, TILE_HEIGHT))
            v5 = cv2.resize(v5, (target_w, TILE_HEIGHT))
            v6 = cv2.resize(v6, (target_w, TILE_HEIGHT))

            # Labels
            add_label(v1, "1. RGB")
            add_label(v2, "2. IR")
            add_label(v3, "3. Event (Raw)")
            add_label(v4, "4. Depth")
            add_label(v5, "5. IR + Event")
            add_label(v6, "6. Depth + Event")

            # Stack: 2 Rows, 3 Cols
            row1 = np.hstack((v1, v2, v3))
            row2 = np.hstack((v4, v5, v6))
            full_screen = np.vstack((row1, row2))

            cv2.imshow("Multi-View Dashboard", full_screen)

            key = cv2.waitKey(1)
            if key == 27: break
            elif key == ord('w'): offset_y -= 1.0
            elif key == ord('s'): offset_y += 1.0
            elif key == ord('a'): offset_x -= 1.0
            elif key == ord('d'): offset_x += 1.0

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()