# Copyright ...
"""
SPACE toggles recording (debounced)
Files saved in dataset/ as 0001_event.raw, 0002_event.raw, ...
"""

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent, UIAction
import argparse
import os


def parse_args():
    return argparse.ArgumentParser().parse_args()


def get_next_filename(output_dir):
    files = [f for f in os.listdir(output_dir) if f.endswith("_event.raw")]
    if not files:
        return "0001_event.raw"
    nums = [int(f[:4]) for f in files if f[:4].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"{next_num:04d}_event.raw"


def run():
    args = parse_args()

    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    device = initiate_device("")
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()

    is_recording = False

    with MTWindow("Metavision Events Viewer", width, height,
                  BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording

            # Only trigger on PRESS to prevent double toggles
            if action != UIAction.PRESS:
                return

            # SPACE toggles recording
            if key == UIKeyEvent.KEY_SPACE:
                if not is_recording:
                    filename = get_next_filename(output_dir)
                    path = os.path.join(output_dir, filename)
                    print(f"🎥 Start recording → {path}")
                    device.get_i_events_stream().log_raw_data(path)
                    is_recording = True
                else:
                    print("⏹ Stop recording.")
                    device.get_i_events_stream().stop_log_raw_data()
                    is_recording = False

            # ESC or Q → quit
            if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height,
            fps=25, palette=ColorPalette.Dark
        )
        event_frame_gen.set_output_callback(
            lambda ts, frame: window.show_async(frame)
        )

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                if is_recording:
                    print("⏹ Stop recording.")
                    device.get_i_events_stream().stop_log_raw_data()
                break


if __name__ == "__main__":
    run()
