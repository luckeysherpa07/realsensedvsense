import os
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

# -------------------------------
# Metavision Viewer
# -------------------------------
def metavision_viewer(file_path):
    """Launch the Metavision event viewer for a given file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    mv_iterator = EventsIterator(input_path=file_path, delta_t=1000)
    height, width = mv_iterator.get_size()

    if not is_live_camera(file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                           fps=25, palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break

# -------------------------------
# Playback Recorded Files
# -------------------------------
def run():
    """List available event files and launch viewer for selected file."""
    dataset_path = "dataset"
    
    files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.raw', '.dat', '.hdf5'))]
    if not files:
        print(f"No event files found in {dataset_path}")
        return

    print("\nAvailable event files:")
    for idx, f in enumerate(files, start=1):
        print(f"{idx}. {f}")

    choice = input(f"\nSelect a file to play (1-{len(files)}): ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
        print("Invalid choice!")
        return

    selected_file = os.path.join(dataset_path, files[int(choice) - 1])
    print(f"\nPlaying file: {selected_file}\n")

    # Launch Metavision viewer
    metavision_viewer(selected_file)
