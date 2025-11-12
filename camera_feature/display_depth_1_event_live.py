import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager

def run():
    # Initialize RealSense pipeline and config
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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

    #Initializing timestamp as 0 for both event and depth
    start_event_timestamp = None
    start_depth_timestamp = None

    try:
        while True:
            # RealSense frame acquisition
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            # Get timestamps for RealSense Depth Camera
            depth_timestamps = depth_frame.get_timestamp()
            print(f"Depth frame timestamp: {depth_timestamps} ms")

            #Mapping the value for Depth Camera
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)


            ###############################
            # DVSense event frame acquisition
            events = camera.get_next_batch()

            # Extract timestamps as a NumPy array
            event_timestamps = events['timestamp'] 
            print("Event timestamps:", event_timestamps)

            #Mappig the value for Event Camera
            histogram = torch.zeros((2, height, width), dtype=torch.long)
            x_coords = torch.tensor(events['x'].astype(np.int32), dtype=torch.long)
            y_coords = torch.tensor(events['y'].astype(np.int32), dtype=torch.long)
            polarities = torch.tensor(events['polarity'].astype(np.int32), dtype=torch.long)
            torch.index_put_(
                histogram, (polarities, y_coords, x_coords), torch.ones_like(x_coords), accumulate=False
            )
            off_histogram, on_histogram = histogram.cpu().numpy()
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:, :] = color_coding['bg']
            canvas[on_histogram > 0] = color_coding['on']
            canvas[off_histogram > 0] = color_coding['off']

            # Display all windows side by side
            cv2.imshow('RealSense Depth', depth_cm)
            cv2.imshow('DVSense Events', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
