#!/usr/bin/env python3
"""
Integrated recorder:
- Metavision DVS events (raw .raw)
- Intel RealSense RGB, Depth, IR at 1280x720 to AVI
Controls:
  SPACE -> toggle start/stop all recordings
  ESC / Q -> quit
Notes:
 - DVS batch window reduced to avoid NonMonotonicTimeHigh
 - RealSense runs in separate thread; poll_for_frames is used
"""

import os
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent, UIAction

# ------------------------------
# Configuration
# ------------------------------
DATASET_FOLDER = "dataset"
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 15
DVS_BATCH_TIME_US = 1000
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"XVID")
VIDEO_FPS = RS_FPS

# ------------------------------
def get_next_index(folder):
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if len(f) >= 4 and f[:4].isdigit()]
    nums = [int(f[:4]) for f in files] if files else []
    return max(nums) + 1 if nums else 1

# ------------------------------
class RealSenseWorker(threading.Thread):
    def __init__(self, pipe, stop_event, shared_state):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.stop_event = stop_event
        self.shared_state = shared_state

    def run(self):
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("IR", cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            frames = self.pipe.poll_for_frames()
            if not frames:
                time.sleep(0.005)
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            if not depth_frame or not color_frame or not ir_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            ir = np.asanyarray(ir_frame.get_data())

            # Depth colormap
            depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

            # Display resized to fit screen
            cv2.imshow("RGB", cv2.resize(color, (960, 540)))
            cv2.imshow("Depth", cv2.resize(depth_colormap, (960, 540)))
            cv2.imshow("IR", cv2.resize(ir_bgr, (960, 540)))

            # Write full-resolution if recording
            if self.shared_state["is_recording"]:
                with self.shared_state["writers_lock"]:
                    rgb_w = self.shared_state.get("rgb_writer")
                    depth_w = self.shared_state.get("depth_writer")
                    ir_w = self.shared_state.get("ir_writer")
                    if rgb_w and depth_w and ir_w:
                        try:
                            rgb_w.write(color)
                            depth_w.write(depth_colormap)
                            ir_w.write(ir_bgr)
                        except Exception as e:
                            print(f"[RealSenseWorker] Write frame error: {e}")

            cv2.waitKey(1)
        # Release windows
        cv2.destroyWindow("RGB")
        cv2.destroyWindow("Depth")
        cv2.destroyWindow("IR")

# ------------------------------
def run():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # Initialize Metavision
    device = initiate_device("")
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()

    # Reduce batch time
    try:
        device.get_i_events_stream().set_batch_events_time(DVS_BATCH_TIME_US)
    except Exception:
        try:
            device.set_batch_events_time(DVS_BATCH_TIME_US)
        except Exception:
            pass

    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                       palette=ColorPalette.Dark)

    # Initialize RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
    cfg.enable_stream(rs.stream.infrared, RS_WIDTH, RS_HEIGHT, rs.format.y8, RS_FPS)
    pipe.start(cfg)

    shared_state = {
        "is_recording": False,
        "current_index": None,
        "rgb_writer": None,
        "depth_writer": None,
        "ir_writer": None,
        "writers_lock": threading.Lock()
    }

    stop_event = threading.Event()
    rs_worker = RealSenseWorker(pipe, stop_event, shared_state)
    rs_worker.start()

    dvs_logging_active = False
    print("\nControls:\n SPACE -> Start/Stop recording\n ESC/Q -> Quit\n")

    def on_cd_frame_cb(ts, cd_frame):
        window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal dvs_logging_active
            if action != UIAction.PRESS:
                return

            if key == UIKeyEvent.KEY_SPACE:
                new_state = not shared_state["is_recording"]
                shared_state["is_recording"] = new_state

                if new_state:
                    idx = get_next_index(DATASET_FOLDER)
                    shared_state["current_index"] = idx

                    # Start DVS
                    event_file = os.path.join(DATASET_FOLDER, f"{idx:04d}_event.raw")
                    try:
                        device.get_i_events_stream().log_raw_data(event_file)
                        dvs_logging_active = True
                        print(f"🎥 DVS recording -> {event_file}")
                    except Exception as e:
                        print(f"[Error] Failed DVS: {e}")
                        shared_state["is_recording"] = False
                        shared_state["current_index"] = None
                        return

                    # RealSense writers
                    with shared_state["writers_lock"]:
                        rgb_path = os.path.join(DATASET_FOLDER, f"{idx:04d}_rgb.avi")
                        depth_path = os.path.join(DATASET_FOLDER, f"{idx:04d}_depth.avi")
                        ir_path = os.path.join(DATASET_FOLDER, f"{idx:04d}_ir.avi")
                        shared_state["rgb_writer"] = cv2.VideoWriter(rgb_path, VIDEO_FOURCC, VIDEO_FPS, (RS_WIDTH, RS_HEIGHT))
                        shared_state["depth_writer"] = cv2.VideoWriter(depth_path, VIDEO_FOURCC, VIDEO_FPS, (RS_WIDTH, RS_HEIGHT))
                        shared_state["ir_writer"] = cv2.VideoWriter(ir_path, VIDEO_FOURCC, VIDEO_FPS, (RS_WIDTH, RS_HEIGHT))
                        print(f"🎥 RealSense recording -> {rgb_path}, {depth_path}, {ir_path}")

                else:
                    print("⏹ Stop all recordings")
                    if dvs_logging_active:
                        try:
                            device.get_i_events_stream().stop_log_raw_data()
                        except Exception:
                            pass
                        dvs_logging_active = False
                    with shared_state["writers_lock"]:
                        if shared_state.get("rgb_writer"): shared_state["rgb_writer"].release()
                        if shared_state.get("depth_writer"): shared_state["depth_writer"].release()
                        if shared_state.get("ir_writer"): shared_state["ir_writer"].release()
                        shared_state["rgb_writer"] = shared_state["depth_writer"] = shared_state["ir_writer"] = None
                    shared_state["current_index"] = None

            if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        try:
            for evs in mv_iterator:
                EventLoop.poll_and_dispatch()
                event_frame_gen.process_events(evs)
                if window.should_close():
                    break
        except Exception as e:
            print(f"[Error] Main loop exception: {e}")

    # Cleanup
    stop_event.set()
    rs_worker.join(timeout=2.0)

    if dvs_logging_active:
        try: device.get_i_events_stream().stop_log_raw_data()
        except Exception: pass

    with shared_state["writers_lock"]:
        if shared_state.get("rgb_writer"): shared_state["rgb_writer"].release()
        if shared_state.get("depth_writer"): shared_state["depth_writer"].release()
        if shared_state.get("ir_writer"): shared_state["ir_writer"].release()

    try: pipe.stop()
    except Exception: pass
    try: device.stop()
    except Exception: pass
    try: cv2.destroyAllWindows()
    except Exception: pass

    print("All streams stopped. Exit.")

# ------------------------------
if __name__ == "__main__":
    run()
