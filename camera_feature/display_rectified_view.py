import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager

# ---------------- Load calibration from YAML ----------------
fs = cv2.FileStorage("stereo_calibration_checkerboard.yaml", cv2.FILE_STORAGE_READ)
mtx_dvs = fs.getNode("DVS_intrinsics").mat()
dist_dvs = fs.getNode("DVS_distortion").mat()
mtx_ir = fs.getNode("IR_intrinsics").mat()
dist_ir = fs.getNode("IR_distortion").mat()
R = fs.getNode("Rotation").mat()
T = fs.getNode("Translation").mat()
fs.release()

# ---------------- Initialize RealSense IR ----------------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipe.start(cfg)

# ---------------- Initialize DVS ----------------
dvs_manager = DvsCameraManager()
dvs_manager.update_cameras()
camera_descs = dvs_manager.get_camera_descs()
if not camera_descs:
    print("No DVS camera found. Exiting...")
    pipe.stop()
    exit(0)

camera = dvs_manager.open_camera(camera_descs[0].serial)
width, height = camera.get_width(), camera.get_height()
camera.start()
camera.set_batch_events_time(10000)

# ---------------- Compute rectification maps ----------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_dvs, dist_dvs,
    mtx_ir, dist_ir,
    (width, height), R, T, alpha=0
)

map1_dvs, map2_dvs = cv2.initUndistortRectifyMap(
    mtx_dvs, dist_dvs, R1, P1, (width, height), cv2.CV_32FC1
)
map1_ir, map2_ir = cv2.initUndistortRectifyMap(
    mtx_ir, dist_ir, R2, P2, (width, height), cv2.CV_32FC1
)

print("Rectification maps computed. Starting streams...")

try:
    while True:
        # ---------------- RealSense IR frame ----------------
        frames = pipe.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1)
        if not ir_frame:
            continue
        ir_image = np.asanyarray(ir_frame.get_data())

        # Rectify IR
        ir_rect = cv2.remap(ir_image, map1_ir, map2_ir, interpolation=cv2.INTER_LINEAR)

        # ---------------- DVS event frame ----------------
        events = camera.get_next_batch()
        if events['x'].size == 0:
            continue

        histogram = np.zeros((height, width), dtype=np.uint8)
        histogram[events['y'], events['x']] = 255  # simple accumulation
        dvs_gray = histogram

        # Apply Gaussian blur for better visualization
        dvs_blur = cv2.GaussianBlur(dvs_gray, (5, 5), 0)

        # Rectify DVS
        dvs_rect = cv2.remap(dvs_blur, map1_dvs, map2_dvs, interpolation=cv2.INTER_LINEAR)

        # Show rectified streams
        cv2.imshow("Rectified DVS", dvs_rect)
        cv2.imshow("Rectified IR", ir_rect)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipe.stop()
    camera.stop()
    cv2.destroyAllWindows()
