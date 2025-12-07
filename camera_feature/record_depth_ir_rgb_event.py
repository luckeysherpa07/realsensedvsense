import numpy as np
import cv2
import os
from dvsense_driver.camera_manager import DvsCameraManager
import pyrealsense2 as rs

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
    # -------------------------------
    # Initialize DVS camera
    # -------------------------------
    dvs_camera_manager = DvsCameraManager()
    dvs_camera_manager.update_cameras()
    camera_descs = dvs_camera_manager.get_camera_descs()
    if not camera_descs:
        print("No DVS camera found.")
        return

    dvs_camera = dvs_camera_manager.open_camera(camera_descs[0].serial)
    dvs_width = dvs_camera.get_width()
    dvs_height = dvs_camera.get_height()
    dvs_camera.start()
    dvs_camera.set_batch_events_time(10000)

    COLOR = {
        'on': [216, 223, 236],
        'off': [201, 126, 64],
        'bg': [0, 0, 0]
    }

    dvs_recording = False
    dataset_folder = "dataset"

    # -------------------------------
    # Initialize RealSense
    # -------------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
    pipe.start(cfg)

    print("\nControls:")
    print(" SPACE = Start/Stop DVS recording")
    print(" Q = Quit\n")

    try:
        while True:
            # -------------------------------
            # DVS events
            # -------------------------------
            events = dvs_camera.get_next_batch()
            canvas = np.zeros((dvs_height, dvs_width, 3), dtype=np.uint8)
            canvas[:, :] = COLOR['bg']

            if events is not None:
                x = events['x'].astype(np.int32)
                y = events['y'].astype(np.int32)
                p = events['polarity'].astype(np.int32)

                valid = (x >= 0) & (x < dvs_width) & (y >= 0) & (y < dvs_height)
                x, y, p = x[valid], y[valid], p[valid]

                canvas[y[p == 1], x[p == 1]] = COLOR['on']
                canvas[y[p == 0], x[p == 0]] = COLOR['off']

                # If recording, save using DVS camera SDK
                if dvs_recording:
                    dvs_camera.start_recording(current_filename)

            if dvs_recording:
                cv2.putText(canvas, "REC", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("DVS Events", canvas)

            # -------------------------------
            # RealSense streams
            # -------------------------------
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)

            if depth_frame and color_frame and ir_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                ir_image = np.asanyarray(ir_frame.get_data())

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.1),
                    cv2.COLORMAP_JET
                )

                cv2.imshow("RGB", color_image)
                cv2.imshow("Depth", depth_colormap)
                cv2.imshow("IR", ir_image)

            # -------------------------------
            # Key handling
            # -------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Toggle DVS recording
                dvs_recording = not dvs_recording
                if dvs_recording:
                    current_filename = get_next_filename(dataset_folder)
                    dvs_camera.start_recording(current_filename)
                    print(f"DVS Recording started → {current_filename}")
                else:
                    dvs_camera.stop_recording()
                    print("DVS Recording stopped.")

            if key == ord('q'):
                break

    finally:
        # Stop everything safely
        if dvs_recording:
            dvs_camera.stop_recording()
        dvs_camera.stop()
        pipe.stop()
        cv2.destroyAllWindows()
        print("All streams stopped. Exit.")


if __name__ == "__main__":
    run()
