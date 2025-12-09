# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Modified sample code:
- Recording starts only when SPACE is pressed
- Files saved in 'dataset/' folder as 0001_event.raw, 0002_event.raw, ...
"""

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()


def get_next_filename(output_dir):
    files = [f for f in os.listdir(output_dir) if f.endswith("_event.raw")]
    if not files:
        return "0001_event.raw"

    nums = [int(f[:4]) for f in files if f[:4].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"{next_num:04d}_event.raw"


def run():
    args = parse_args()

    # Force output directory to 'dataset'
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    # HAL Device on live camera
    device = initiate_device("")

    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()

    is_recording = False
    log_path = None

    # Window
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, log_path

            # SPACE → Start recording
            if key == UIKeyEvent.KEY_SPACE and not is_recording:
                filename = get_next_filename(output_dir)
                log_path = os.path.join(output_dir, filename)
                print(f"🎥 Start recording → {log_path}")
                device.get_i_events_stream().log_raw_data(log_path)
                is_recording = True

            # ESC or Q → Stop & exit
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event frame generator for visualization
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=25,
            palette=ColorPalette.Dark
        )

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Main loop
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                if is_recording:
                    print("🛑 Stop recording.")
                    device.get_i_events_stream().stop_log_raw_data()
                break


if __name__ == "__main__":
    run()
