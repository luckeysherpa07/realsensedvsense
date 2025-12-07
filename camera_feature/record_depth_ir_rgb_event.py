import numpy as np
import cv2
import os
from dvsense_driver.camera_manager import DvsCameraManager


def get_next_filename(folder):
    os.makedirs(folder, exist_ok=True)

    existing_files = [
        f for f in os.listdir(folder)
        if f.startswith("event_") and f.endswith(".aedat4")
    ]

    if not existing_files:
        return os.path.join(folder, "event_0001.aedat4")

    numbers = [int(f[6:10]) for f in existing_files]
    next_num = max(numbers) + 1
    return os.path.join(folder, f"event_{next_num:04d}.aedat4")


def run():
    dvs_camera_manager = DvsCameraManager()
    dvs_camera_manager.update_cameras()

    camera_descs = dvs_camera_manager.get_camera_descs()
    if not camera_descs:
        print("No camera found.")
        return

    camera = dvs_camera_manager.open_camera(camera_descs[0].serial)
    width = camera.get_width()
    height = camera.get_height()

    camera.start()
    camera.set_batch_events_time(10000)

    COLOR = {
        'on': [216, 223, 236],
        'off': [201, 126, 64],
        'bg': [0, 0, 0]
    }

    recording = False
    folder = "dataset"

    print("\nControls:")
    print(" SPACE = Start/Stop recording")
    print(" Q = Quit\n")

    while True:
        events = camera.get_next_batch()
        if events is None:
            continue

        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['polarity'].astype(np.int32)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, p = x[valid], y[valid], p[valid]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = COLOR['bg']

        # On/Off event colors (fixed indexing!)
        canvas[y[p == 1], x[p == 1]] = COLOR['on']
        canvas[y[p == 0], x[p == 0]] = COLOR['off']

        if recording:
            cv2.putText(canvas, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("DVS Events", canvas)
        key = cv2.waitKey(1) & 0xFF

        # Toggle recording on SPACE
        if key == ord(' '):
            recording = not recording
            if recording:
                filename = get_next_filename(folder)
                camera.start_recording(filename)
                print(f"Recording started → {filename}")
            else:
                camera.stop_recording()
                print("Recording stopped.")

        if key == ord('q'):
            break

    if recording:
        camera.stop_recording()

    camera.stop()
    cv2.destroyAllWindows()
    print("Camera stopped. Exit.")


if __name__ == "__main__":
    run()
