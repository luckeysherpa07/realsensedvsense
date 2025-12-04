import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager

def run():
    # ---------------- Load calibration using OpenCV FileStorage ----------------
    fs = cv2.FileStorage("stereo_calibration_checkerboard.yaml", cv2.FILE_STORAGE_READ)
    
    IR_K = fs.getNode("IR_intrinsics").mat()
    IR_dist = fs.getNode("IR_distortion").mat().flatten()
    DVS_K = fs.getNode("DVS_intrinsics").mat()
    DVS_dist = fs.getNode("DVS_distortion").mat().flatten()
    R = fs.getNode("Rotation").mat()
    T = fs.getNode("Translation").mat()
    fs.release()

    # Image sizes
    IR_size = (1280, 720)
    DVS_size = (1280, 720)

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        IR_K, IR_dist,
        DVS_K, DVS_dist,
        IR_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    IR_map1, IR_map2 = cv2.initUndistortRectifyMap(IR_K, IR_dist, R1, P1, IR_size, cv2.CV_32FC1)
    DVS_map1, DVS_map2 = cv2.initUndistortRectifyMap(DVS_K, DVS_dist, R2, P2, DVS_size, cv2.CV_32FC1)

    # ---------------- Initialize cameras ----------------
    # RealSense IR
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, IR_size[0], IR_size[1], rs.format.y8, 30)
    pipe.start(cfg)

    # DVSense
    dvs_camera_manager = DvsCameraManager()
    dvs_camera_manager.update_cameras()
    camera_descriptions = dvs_camera_manager.get_camera_descs()
    if not camera_descriptions:
        print("No DVS camera found")
        pipe.stop()
        return

    camera = dvs_camera_manager.open_camera(camera_descriptions[0].serial)
    width = camera.get_width()
    height = camera.get_height()
    COLOR_CODING = {'blue_white': {'on':[216,223,236], 'off':[201,126,64], 'bg':[0,0,0]}}
    color_coding = COLOR_CODING['blue_white']
    camera.start()
    camera.set_batch_events_time(10000)

    try:
        while True:
            # ---------------- RealSense IR ----------------
            frames = pipe.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue
            ir_image = np.asanyarray(ir_frame.get_data())
            ir_rect = cv2.remap(ir_image, IR_map1, IR_map2, cv2.INTER_LINEAR)

            # ---------------- DVS events ----------------
            events = camera.get_next_batch()
            if events['x'].size == 0:
                continue

            histogram = torch.zeros((2, height, width), dtype=torch.long)
            x_coords = torch.tensor(events['x'].astype(np.int32), dtype=torch.long)
            y_coords = torch.tensor(events['y'].astype(np.int32), dtype=torch.long)
            polarities = torch.tensor(events['polarity'].astype(np.int32), dtype=torch.long)
            torch.index_put_(histogram, (polarities, y_coords, x_coords), torch.ones_like(x_coords), accumulate=True)
            off_hist, on_hist = histogram.cpu().numpy()
            event_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            event_canvas[:, :] = color_coding['bg']
            event_canvas[on_hist>0] = color_coding['on']
            event_canvas[off_hist>0] = color_coding['off']

            # Rectify DVS
            event_rect = cv2.remap(event_canvas, DVS_map1, DVS_map2, cv2.INTER_LINEAR)

            # ---------------- Overlay ----------------
            overlay = cv2.addWeighted(cv2.cvtColor(ir_rect, cv2.COLOR_GRAY2BGR), 0.5, event_rect, 0.5, 0)

            # Display
            cv2.imshow("IR Rectified", ir_rect)
            cv2.imshow("DVS Rectified", event_rect)
            cv2.imshow("Overlay", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
