#!/usr/bin/env python3
"""
Integrated recorder:
- Metavision DVS events (raw .raw) with preview
- Intel RealSense RGB, Depth, IR at 1280x720 to .bag (no preview)
Controls:
  SPACE -> toggle start/stop all recordings
  ESC / Q -> quit
"""

import os
import threading
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

# ------------------------------
def get_next_index(folder):
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if len(f) >= 4 and f[:4].isdigit()]
    nums = [int(f[:4]) for f in files] if files else []
    return max(nums) + 1 if nums else 1

# ------------------------------
def run():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # ------------------------------
    # Initialize Metavision DVS
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

    shared_state = {
        "is_recording": False,
        "current_index": None,
        "rs_pipe": None,
        "rs_cfg": None,
        "rs_bag_path": None,
        "dvs_logging_active": False
    }

    print("\nControls:\n SPACE -> Start/Stop recording\n ESC/Q -> Quit\n")

    def on_cd_frame_cb(ts, cd_frame):
        window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.PRESS:
                return

            if key == UIKeyEvent.KEY_SPACE:
                new_state = not shared_state["is_recording"]
                shared_state["is_recording"] = new_state

                if new_state:
                    idx = get_next_index(DATASET_FOLDER)
                    shared_state["current_index"] = idx

                    # ------------------------------
                    # Start DVS recording
                    event_file = os.path.join(DATASET_FOLDER, f"{idx:04d}_event.raw")
                    try:
                        device.get_i_events_stream().log_raw_data(event_file)
                        shared_state["dvs_logging_active"] = True
                        print(f"🎥 DVS recording -> {event_file}")
                    except Exception as e:
                        print(f"[Error] Failed DVS recording: {e}")
                        shared_state["is_recording"] = False
                        shared_state["current_index"] = None
                        return

                    # ------------------------------
                    # Start RealSense .bag recording
                    bag_path = os.path.join(DATASET_FOLDER, f"{idx:04d}_realsense.bag")
                    try:
                        pipe = rs.pipeline()
                        cfg = rs.config()
                        cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
                        cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
                        cfg.enable_stream(rs.stream.infrared, RS_WIDTH, RS_HEIGHT, rs.format.y8, RS_FPS)
                        cfg.enable_record_to_file(bag_path)
                        pipe.start(cfg)
                        shared_state["rs_pipe"] = pipe
                        shared_state["rs_cfg"] = cfg
                        shared_state["rs_bag_path"] = bag_path
                        print(f"🎥 RealSense recording -> {bag_path}")
                    except Exception as e:
                        print(f"[Error] Failed RealSense .bag recording: {e}")
                        shared_state["is_recording"] = False
                        shared_state["current_index"] = None
                        shared_state["dvs_logging_active"] = False
                        return

                else:
                    print("⏹ Stop all recordings")

                    # Stop DVS
                    if shared_state["dvs_logging_active"]:
                        try:
                            device.get_i_events_stream().stop_log_raw_data()
                        except Exception:
                            pass
                        shared_state["dvs_logging_active"] = False

                    # Stop RealSense recording
                    try:
                        if shared_state["rs_pipe"]:
                            shared_state["rs_pipe"].stop()
                            shared_state["rs_pipe"] = None
                            shared_state["rs_cfg"] = None
                            shared_state["rs_bag_path"] = None
                    except Exception:
                        pass

                    shared_state["current_index"] = None

            elif key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
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

    # ------------------------------
    # Cleanup
    try:
        if shared_state["rs_pipe"]:
            shared_state["rs_pipe"].stop()
    except Exception:
        pass
    try:
        if shared_state["dvs_logging_active"]:
            device.get_i_events_stream().stop_log_raw_data()
    except Exception:
        pass
    try:
        device.stop()
    except Exception:
        pass
    try:
        window.destroy()
    except Exception:
        pass

    print("All streams stopped. Exit.")


# ------------------------------
if __name__ == "__main__":
    run()
