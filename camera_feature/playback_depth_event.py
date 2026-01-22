import os
import cv2
import time  # <--- Added for manual sync
import numpy as np
import pyrealsense2 as rs
import torch
from metavision_core.event_io import EventsIterator

# ---------------- CONFIGURATION ----------------
DEPTH_SCALE_FACTOR = 100.0   
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TILE_HEIGHT = 400 
VISUALIZATION_WINDOW_US = 30000  # Fix exposure to 30ms

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
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img

def run():
    # --- PATHS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(parent_dir, "dataset")
    calib_file = os.path.join(parent_dir, "stereo_calibration_checkerboard.yaml")
    
    if not os.path.exists(dataset_path): 
        print("Dataset not found"); return

    files = os.listdir(dataset_path)
    prefixes = sorted(list(set([f.replace("_event.raw", "") for f in files if f.endswith("_event.raw")])))
    if not prefixes: 
        print("No event files found"); return

    print("\nAvailable recordings:")
    for idx, prefix in enumerate(prefixes, start=1): 
        print(f"{idx}. {prefix}")
    choice = input(f"\nSelect (1-{len(prefixes)}): ").strip()
    if not choice.isdigit() or int(choice) < 1: return
    selected_prefix = prefixes[int(choice) - 1]

    event_file = os.path.join(dataset_path, f"{selected_prefix}_event.raw")
    bag_file = os.path.join(dataset_path, f"{selected_prefix}_realsense.bag")
    delta_file = os.path.join(dataset_path, f"{selected_prefix}_delta_seconds.txt")

    # --- LOAD CALIB + DELTA ---
    calib_data = load_calibration_yaml(calib_file)
    delta_us = 0
    if os.path.exists(delta_file):
        try:
            with open(delta_file, "r") as f: 
                delta_us = int(float(f.read().strip()) * 1e6)
        except: pass

    # --- RESOURCES ---
    def reset_event_iterator():
        it = EventsIterator(input_path=event_file, delta_t=10000)
        h, w = it.get_size()
        return it, iter(it), h, w

    mv_iterator, ev_iter, ev_h, ev_w = reset_event_iterator()
    dvs_hist_gpu = torch.zeros((2, ev_h, ev_w), dtype=torch.float32, device=DEVICE)

    # --- REALSENSE ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=True)
    profile = pipeline.start(config)
    
    dev = profile.get_device()
    playback = dev.as_playback()
    
    # [FIX 1] Disable internal real-time clock to prevent buffering/skipping
    playback.set_real_time(False) 

    rs_start_ts_ms = None
    prev_ts_ms = -1
    leftover_events = None
    ev_start_ts_us = None

    print("\n[Controls] ESC: Quit")

    try:
        while True:
            # [FIX 2] Start measuring processing time
            iter_start_time = time.time()

            try:
                # With real_time=False, this returns immediately with the next sequential frame
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError: 
                break

            depth_frame = frames.get_depth_frame()
            if not depth_frame: 
                continue
            
            # --- SYNC TIME ---
            curr_ms = frames.get_timestamp()
            
            # Determine how much time this frame *should* take (Video Delta)
            # Default to 33ms (30fps) if it's the first frame
            video_dt_sec = (curr_ms - prev_ts_ms) / 1000.0 if prev_ts_ms > 0 else 0.033
            
            # Handle Loop/Reset
            if prev_ts_ms > 0 and curr_ms < prev_ts_ms:
                mv_iterator, ev_iter, _, _ = reset_event_iterator()
                leftover_events = None
                ev_start_ts_us = None
                rs_start_ts_ms = curr_ms 
                video_dt_sec = 0.033 # Reset delta on loop
                
            prev_ts_ms = curr_ms
            if rs_start_ts_ms is None: 
                rs_start_ts_ms = curr_ms
                
            elapsed_us = int((curr_ms - rs_start_ts_ms) * 1000)

            # --- DEPTH ---
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.08), cv2.COLORMAP_JET)

            # --- EVENTS ---
            dvs_hist_gpu.zero_()
            if ev_iter:
                while ev_start_ts_us is None:
                    if leftover_events is not None and len(leftover_events) > 0:
                        ev_start_ts_us = leftover_events['t'][0]; break
                    try: 
                        leftover_events = next(ev_iter)
                    except StopIteration: 
                        ev_iter = None; break
                
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
                            except StopIteration: 
                                ev_iter = None; break
                        
                        if accumulated:
                            all_evs = np.concatenate(accumulated)
                            idx = np.searchsorted(all_evs['t'], target_ev, side='right')
                            events_now, leftover_events = all_evs[:idx], all_evs[idx:]
                            
                            if len(events_now) > 0:
                                t_viz_start = target_ev - VISUALIZATION_WINDOW_US
                                idx_viz = np.searchsorted(events_now['t'], t_viz_start, side='left')
                                events_visual = events_now[idx_viz:]

                                if len(events_visual) > 0:
                                    xs = torch.from_numpy(events_visual['x'].astype(np.int64)).to(DEVICE)
                                    ys = torch.from_numpy(events_visual['y'].astype(np.int64)).to(DEVICE)
                                    ps = torch.from_numpy(events_visual['p'].astype(np.int64)).to(DEVICE)
                                    
                                    mask_ev = (xs < ev_w) & (ys < ev_h)
                                    if mask_ev.any():
                                        dvs_hist_gpu.index_put_((ps[mask_ev], ys[mask_ev], xs[mask_ev]), 
                                                               torch.tensor(1.0, device=DEVICE), accumulate=True)

            # --- EVENT IMAGE ---
            dvs_raw_clr = torch.stack((dvs_hist_gpu[0], torch.zeros_like(dvs_hist_gpu[0]), dvs_hist_gpu[1]), dim=0)
            if dvs_raw_clr.max() > 0: 
                dvs_raw_clr = torch.clamp(dvs_raw_clr / 5.0, 0, 1)
            event_img = (dvs_raw_clr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # --- RESIZE + HSTACK ---
            event_resized = resize_maintain_aspect(event_img, TILE_HEIGHT)
            depth_resized = resize_maintain_aspect(depth_colormap, TILE_HEIGHT)

            target_w = max(event_resized.shape[1], depth_resized.shape[1])
            event_resized = cv2.resize(event_resized, (target_w, TILE_HEIGHT))
            depth_resized = cv2.resize(depth_resized, (target_w, TILE_HEIGHT))

            event_resized = add_label(event_resized, "EVENTS")
            depth_resized = add_label(depth_resized, "DEPTH")

            combined = np.hstack((event_resized, depth_resized))
            cv2.imshow("Event | Depth", combined)

            # [FIX 3] Manual Sync Logic
            # Calculate how long the loop took
            process_duration = time.time() - iter_start_time
            
            # If we processed faster than the video frame rate, sleep the difference
            if process_duration < video_dt_sec:
                time.sleep(video_dt_sec - process_duration)
            
            # If process_duration > video_dt_sec, we simply loop immediately.
            # This results in smooth "slow motion" rather than jittery "catch up".

            if cv2.waitKey(1) == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()