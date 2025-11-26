import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager

def run():
    # Initialize RealSense pipeline and config
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipe.start(cfg)

    # Initialize DVSense event camera
    dvs_camera_manager = DvsCameraManager()
    dvs_camera_manager.update_cameras()
    camera_descriptions = dvs_camera_manager.get_camera_descs()
    if not camera_descriptions:
        print("No DVS camera found. Exiting...")
        pipe.stop()
        exit(0)

    camera = dvs_camera_manager.open_camera(camera_descriptions[0].serial)
    width = camera.get_width()
    height = camera.get_height()
    COLOR_CODING = {
        'blue_white': {
            'on': [216, 223, 236],
            'off': [201, 126, 64],
            'bg': [0, 0, 0]
        }
    }
    color_coding = COLOR_CODING['blue_white']
    camera.start()
    camera.set_batch_events_time(10000)  # 10 ms batch for event camera

    # Initialize timestamps
    start_event_timestamp = None
    start_depth_timestamp = None

    try:
        while True:
            # ---------------- RealSense frame acquisition ----------------
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()  # Get RGB frame

            if not depth_frame or not color_frame:
                continue

            # Depth processing
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_canvas = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.1), cv2.COLORMAP_JET)

            # RGB processing
            rgb_image = np.asanyarray(color_frame.get_data())  # Already in BGR format

            # Depth timestamp
            depth_timestamp = depth_frame.get_timestamp()
            print(f"Depth frame timestamp: {depth_timestamp} ms")

            # ---------------- DVSense event frame acquisition ----------------
            events = camera.get_next_batch()
            if events['x'].size == 0:
                continue

            # Event timestamps
            event_timestamps = events['timestamp']
            print("Event timestamps:", event_timestamps)

            # Create histogram
            histogram = torch.zeros((2, height, width), dtype=torch.long)
            x_coords = torch.tensor(events['x'].astype(np.int32), dtype=torch.long)
            y_coords = torch.tensor(events['y'].astype(np.int32), dtype=torch.long)
            polarities = torch.tensor(events['polarity'].astype(np.int32), dtype=torch.long)
            torch.index_put_(
                histogram, (polarities, y_coords, x_coords), torch.ones_like(x_coords), accumulate=True
            )

            off_histogram, on_histogram = histogram.cpu().numpy()
            event_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            event_canvas[:, :] = color_coding['bg']
            event_canvas[on_histogram > 0] = color_coding['on']
            event_canvas[off_histogram > 0] = color_coding['off']

            # ---------------- Timestamp alignment ----------------
            if start_event_timestamp is None:
                start_event_timestamp = event_timestamps[0]
            if start_depth_timestamp is None:
                start_depth_timestamp = depth_timestamp

            depth_timestamp_us = depth_timestamp * 1000
            start_depth_timestamp_us = start_depth_timestamp * 1000
            aligned_event_timestamps = event_timestamps - start_event_timestamp
            aligned_depth_timestamp = depth_timestamp_us - start_depth_timestamp_us

            # Debug prints
            print("Aligned first 10 event timestamps (μs):", aligned_event_timestamps[:10])
            print("Aligned depth frame timestamp (μs):", aligned_depth_timestamp)

            # ---------------- Display ----------------
            cv2.imshow('RealSense RGB', rgb_image)
            cv2.imshow('RealSense Depth', depth_canvas)
            cv2.imshow('DVSense Events', event_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
