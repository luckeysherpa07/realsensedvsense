#!/usr/bin/env python3
"""
Integrated recorder:
- Metavision DVS events (raw .raw) with preview
- Intel RealSense RGB, Depth, IR at 1280x720 to .bag (no preview)
- **IR Emitter Disabled** to prevent noise on the Event Camera

Controls:
  SPACE -> toggle start/stop all recordings
  ESC / Q -> quit
"""

import os
import threading
import time
import pyrealsense2 as rs
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent, UIAction

# ------------------------------
# Configuration
# ------------------------------
DATASET_FOLDER = "dataset"

# RealSense Config
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 30  # Common FPS for D435/D455 at 720p. Use 15 if bandwidth is tight.

# DVS Config
DVS_BATCH_TIME_US = 10000  # 10ms batch time for smoother UI

# ------------------------------
def get_next_index(folder):
    """
    Scans the folder for existing files (0001_*, 0002_*) and returns the next available index.
    """
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if len(f) >= 4 and f[:4].isdigit()]
    nums = [int(f[:4]) for f in files] if files else []
    return max(nums) + 1 if nums else 1

# ------------------------------
# Background Thread for RealSense
# ------------------------------
def realsense_recorder_thread(bag_path, stop_event, error_event):
    """
    Runs the RealSense pipeline in a separate thread.
    1. Configures streams & file recording.
    2. Starts pipeline.
    3. **Disables IR Emitter**.
    4. Loops `wait_for_frames` to pump data into the .bag file.
    """
    pipe = rs.pipeline()
    cfg = rs.config()

    # Configure Streams
    # Note: Ensure your USB connection supports this bandwidth.
    cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
    # IR Index 1 is usually the Left IR camera (aligned with depth origin)
    cfg.enable_stream(rs.stream.infrared, 1, RS_WIDTH, RS_HEIGHT, rs.format.y8, RS_FPS)

    # Enable recording to bag file
    cfg.enable_record_to_file(bag_path)

    try:
        # Start pipeline and get the active profile
        pipeline_profile = pipe.start(cfg)
        print(f"🎥 RealSense started -> {bag_path}")

        # --- DISABLE IR EMITTER ---
        # This removes the dot pattern that confuses the Event Camera
        try:
            device = pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 0.0) # 0 = Off, 1 = On
                print("🚫 RealSense IR Emitter turned OFF successfully.")
            else:
                print("[Warning] This RealSense device does not support disabling the emitter.")
        except Exception as e:
            print(f"[Warning] Could not disable emitter: {e}")
        # ---------------------------

        # Main Loop: Pump frames
        while not stop_event.is_set():
            # blocking call; needed to process frames and write them to disk
            pipe.wait_for_frames(timeout_ms=5000)

    except Exception as e:
        print(f"[Error] RealSense Thread crashed: {e}")
        error_event.set()
    finally:
        try:
            pipe.stop()
            print("⏹ RealSense pipeline stopped.")
        except Exception:
            pass

# ------------------------------
# Main Application
# ------------------------------
def run():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # 1. Initialize Metavision DVS
    print("--- Initializing Metavision Device ---")
    try:
        device = initiate_device("")
    except Exception as e:
        print(f"Error: Could not open Metavision device: {e}")
        return

    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    print(f"DVS Resolution: {width}x{height}")

    # Set batch time
    try:
        device.get_i_events_stream().set_batch_events_time(DVS_BATCH_TIME_US)
    except Exception:
        pass

    # Visualization Algo
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, 
        fps=25,
        palette=ColorPalette.Dark
    )

    # Shared State
    shared_state = {
        "is_recording": False,
        "rs_thread": None,
        "rs_stop_event": threading.Event(),
        "rs_error_event": threading.Event(),
        "dvs_logging_active": False,
        "current_idx": 0
    }

    print("\n" + "="*40)
    print(" CONTROLS:")
    print(" SPACE  -> Start / Stop Recording")
    print(" ESC/Q  -> Quit")
    print("="*40 + "\n")

    def on_cd_frame_cb(ts, cd_frame):
        window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.PRESS:
                return

            if key == UIKeyEvent.KEY_SPACE:
                # --- TOGGLE RECORDING ---
                if not shared_state["is_recording"]:
                    # START
                    idx = get_next_index(DATASET_FOLDER)
                    shared_state["current_idx"] = idx
                    print(f"\n>>> STARTING RECORDING SET {idx:04d} <<<")

                    # A. Start DVS Recording
                    event_file = os.path.join(DATASET_FOLDER, f"{idx:04d}_event.raw")
                    try:
                        device.get_i_events_stream().log_raw_data(event_file)
                        shared_state["dvs_logging_active"] = True
                        print(f"   [DVS] Recording to {event_file}")
                    except Exception as e:
                        print(f"   [Error] DVS failed to start logging: {e}")
                        return

                    # B. Start RealSense Recording (Threaded)
                    bag_path = os.path.join(DATASET_FOLDER, f"{idx:04d}_realsense.bag")
                    shared_state["rs_stop_event"].clear()
                    shared_state["rs_error_event"].clear()
                    
                    t = threading.Thread(target=realsense_recorder_thread, 
                                         args=(bag_path, shared_state["rs_stop_event"], shared_state["rs_error_event"]))
                    t.start()
                    shared_state["rs_thread"] = t
                    
                    shared_state["is_recording"] = True

                else:
                    # STOP
                    print(f"\n<<< STOPPING RECORDING SET {shared_state['current_idx']:04d} <<<")

                    # A. Stop DVS
                    if shared_state["dvs_logging_active"]:
                        device.get_i_events_stream().stop_log_raw_data()
                        shared_state["dvs_logging_active"] = False
                        print("   [DVS] Logging stopped")

                    # B. Stop RealSense
                    if shared_state["rs_thread"] and shared_state["rs_thread"].is_alive():
                        shared_state["rs_stop_event"].set()
                        shared_state["rs_thread"].join()
                        shared_state["rs_thread"] = None
                        # Message is printed by thread

                    shared_state["is_recording"] = False

            elif key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # --- MAIN LOOP ---
        try:
            for evs in mv_iterator:
                # 1. Process Events
                EventLoop.poll_and_dispatch()
                event_frame_gen.process_events(evs)

                # 2. Monitor RealSense Health
                if shared_state["is_recording"] and shared_state["rs_error_event"].is_set():
                    print("!!! CRITICAL ERROR: RealSense thread failed. Stopping all recordings.")
                    # Force stop DVS
                    if shared_state["dvs_logging_active"]:
                        device.get_i_events_stream().stop_log_raw_data()
                        shared_state["dvs_logging_active"] = False
                    shared_state["is_recording"] = False

                # 3. Check Window Close
                if window.should_close():
                    break
        except Exception as e:
            print(f"[Error] Main loop exception: {e}")

    # ------------------------------
    # Cleanup / Shutdown
    # ------------------------------
    print("\nShutting down...")
    if shared_state["is_recording"]:
        if shared_state["dvs_logging_active"]:
            device.get_i_events_stream().stop_log_raw_data()
        
        if shared_state["rs_thread"] and shared_state["rs_thread"].is_alive():
            shared_state["rs_stop_event"].set()
            shared_state["rs_thread"].join()

    device.stop()
    window.destroy()
    print("Done.")

if __name__ == "__main__":
    run()