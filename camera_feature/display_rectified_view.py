#!/usr/bin/env python3
"""
Rectified stereo view for DVS + RealSense IR using a previously saved YAML
containing: DVS_intrinsics, DVS_distortion, IR_intrinsics, IR_distortion,
Rotation, Translation, Essential, Fundamental.

Usage:
    python rectified_view.py
"""
import sys
import os
import numpy as np
import cv2
import pyrealsense2 as rs
from dvsense_driver.camera_manager import DvsCameraManager
import time

YAML_PATH = "stereo_calibration_checkerboard.yaml"
EVENT_BATCH_TIME = 10000  # microseconds (as in your capture script)
COLOR_CODING = {
    'on': [216, 223, 236],
    'off': [201, 126, 64],
    'bg': [0, 0, 0]
}


def load_calibration(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Unable to open calibration file: {yaml_path}")

    mtx_dvs = fs.getNode("DVS_intrinsics").mat()
    dist_dvs = fs.getNode("DVS_distortion").mat()
    mtx_ir = fs.getNode("IR_intrinsics").mat()
    dist_ir = fs.getNode("IR_distortion").mat()
    R = fs.getNode("Rotation").mat()
    T = fs.getNode("Translation").mat()
    E = fs.getNode("Essential").mat()
    F = fs.getNode("Fundamental").mat()

    fs.release()

    if mtx_dvs is None or mtx_ir is None or R is None or T is None:
        raise RuntimeError("One or more required matrices not found in YAML.")

    # Ensure shapes are (3,3) for intrinsics and (3,1) or (1,3) for T
    T = T.reshape(3, 1) if T.size == 3 else T

    return mtx_dvs, dist_dvs, mtx_ir, dist_ir, R, T, E, F


def events_to_frame(events, width, height, color_coding=COLOR_CODING):
    """
    Build an RGB frame from DVS events using numpy (no torch).
    events: dict-like with 'x','y','polarity' arrays (numpy arrays)
    """
    if events['x'].size == 0:
        # Return an empty black image
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = color_coding['bg']
        return frame

    on_hist = np.zeros((height, width), dtype=np.uint32)
    off_hist = np.zeros((height, width), dtype=np.uint32)

    xs = events['x'].astype(np.int32)
    ys = events['y'].astype(np.int32)
    ps = events['polarity'].astype(np.int8)

    # Clip coordinates just in case
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    # accumulate counts
    # faster than for-loop: use np.add.at
    mask_on = ps > 0
    if np.any(mask_on):
        np.add.at(on_hist, (ys[mask_on], xs[mask_on]), 1)
    mask_off = ~mask_on
    if np.any(mask_off):
        np.add.at(off_hist, (ys[mask_off], xs[mask_off]), 1)

    dvs_frame = np.zeros((height, width, 3), dtype=np.uint8)
    dvs_frame[:, :] = color_coding['bg']
    dvs_frame[on_hist > 0] = color_coding['on']
    dvs_frame[off_hist > 0] = color_coding['off']

    return dvs_frame


def create_rectify_maps(mtx1, dist1, mtx2, dist2, R, T, size1, size2):
    """
    Compute stereo rectification and init undistort/rectify maps for both cameras.

    mtx1, dist1: camera1 camera matrix and distortion (DVS)
    mtx2, dist2: camera2 camera matrix and distortion (IR)
    R, T: rotation and translation (from camera1 -> camera2)
    size1, size2: (width, height) tuples for camera1 and camera2
    """
    # stereoRectify expects imageSize (w,h) common to both images; pick a common size.
    # Choose the max so nothing is smaller than maps.
    common_w = max(size1[0], size2[0])
    common_h = max(size1[1], size2[1])
    common_size = (common_w, common_h)

    print("Size1", size1)
    print("Size2", size2)
    print("Common Size", common_size)

    flags = cv2.CALIB_ZERO_DISPARITY
    # alpha = 0 => crop, alpha = -1 => keep all pixels. Use 0 to remove black borders.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, common_size, R, T, flags=flags, alpha=0
    )

    print("R1: ", R1)
    print("R2: ", R2)
    print("P1: ", P1)
    print("P2: ", P2)

    # For each camera, create maps sized to that camera's resolution
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        mtx1, dist1, R1, P1, (size1[0], size1[1]), cv2.CV_32FC1
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        mtx2, dist2, R2, P2, (size2[0], size2[1]), cv2.CV_32FC1
    )

    print("map1 range:", np.min(map1_x), np.max(map1_x))
    print("map2 range:", np.min(map2_x), np.max(map2_x))

    return (map1_x, map1_y), (map2_x, map2_y), Q, (R1, R2, P1, P2)


def run():
    print("Loading calibration...")
    try:
        mtx_dvs, dist_dvs, mtx_ir, dist_ir, R, T, E, F = load_calibration(YAML_PATH)
    except Exception as e:
        print("ERROR loading calibration:", e)
        sys.exit(1)

    # ----------------- Initialize RealSense -----------------
    pipe = rs.pipeline()
    cfg = rs.config()
    # IR stream (index 1) as in your capture script, 1280x720 @30
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipe.start(cfg)
    time.sleep(0.1)

    # get IR frame size (might be 1280x720)
    frames = pipe.wait_for_frames()
    ir_frame = frames.get_infrared_frame(1)
    if not ir_frame:
        print("Could not fetch IR frame from RealSense. Exiting.")
        pipe.stop()
        sys.exit(1)
    ir_image = np.asanyarray(ir_frame.get_data())
    ir_h, ir_w = ir_image.shape

    # ----------------- Initialize DVS camera -----------------
    dvs_manager = DvsCameraManager()
    dvs_manager.update_cameras()
    camera_descs = dvs_manager.get_camera_descs()
    if not camera_descs:
        print("No DVS camera found. Exiting...")
        pipe.stop()
        sys.exit(1)

    camera = dvs_manager.open_camera(camera_descs[0].serial)
    width, height = camera.get_width(), camera.get_height()
    camera.start()
    camera.set_batch_events_time(EVENT_BATCH_TIME)

    print(f"DVS resolution: {width}x{height}, IR resolution: {ir_w}x{ir_h}")

    # ----------------- Build rectification maps -----------------
    try:
        # sizes passed as (width, height)
        maps_dvs, maps_ir, Q, (R1, R2, P1, P2) = create_rectify_maps(
            mtx_dvs, dist_dvs, mtx_ir, dist_ir, R, T, (width, height), (ir_w, ir_h)
        )
    except Exception as e:
        print("StereoRectify / map creation failed:", e)
        camera.stop()
        pipe.stop()
        sys.exit(1)

    map_dvs_x, map_dvs_y = maps_dvs
    map_ir_x, map_ir_y = maps_ir

    print("Rectification maps created. Starting display...")

    save_idx = 0
    try:
        while True:
            # RealSense IR
            frames = pipe.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                # skip iteration if no IR frame
                continue
            ir_image = np.asanyarray(ir_frame.get_data())

            # DVS events
            events = camera.get_next_batch()
            # events is expected to be dict-like with keys 'x','y','polarity'
            # If your driver provides a different structure adapt accordingly.
            if events['x'].size == 0:
                dvs_raw = np.zeros((height, width, 3), dtype=np.uint8)
                dvs_raw[:, :] = COLOR_CODING['bg']
            else:
                dvs_raw = events_to_frame(events, width, height)

            # convert to grayscale + blur as earlier (helps visual quality)
            dvs_gray = cv2.cvtColor(dvs_raw, cv2.COLOR_BGR2GRAY)
            dvs_blur = cv2.GaussianBlur(dvs_gray, (5, 5), 0)
            dvs_for_rect = cv2.cvtColor(dvs_blur, cv2.COLOR_GRAY2BGR)

            # Undistort+rectify (remap)
            rectified_dvs = cv2.remap(dvs_for_rect, map_dvs_x, map_dvs_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # IR is single-channel; convert to BGR for consistent display
            ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            rectified_ir = cv2.remap(ir_bgr, map_ir_x, map_ir_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # ---------------------------------------------------------
            # OVERLAY VIEW
            # ---------------------------------------------------------
            alpha = 0.5  # adjust between 0 (all IR) and 1 (all DVS)
            overlay = cv2.addWeighted(rectified_dvs, alpha, rectified_ir, 1 - alpha, 0)

            # optional: draw epipolar lines on overlay too
            overlay_with_lines = overlay.copy()
            h = overlay_with_lines.shape[0]
            for y in range(0, h, 50):
                cv2.line(overlay_with_lines, (0, y), (overlay_with_lines.shape[1], y), (0, 255, 0), 1)

            cv2.imshow("Overlay (DVS + IR)", overlay_with_lines)

            # Stack horizontally for side-by-side view and draw horizontal guide lines
            combined = np.hstack((
                cv2.resize(rectified_dvs, (rectified_ir.shape[1], rectified_ir.shape[0])),
                rectified_ir
            ))

            # Draw epipolar guide lines every 50 px
            h = combined.shape[0]
            for y in range(0, h, 50):
                cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

            cv2.imshow("Rectified DVS (left) | Rectified IR (right) - Press q to quit, s to save", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # Save rectified pair
                cv2.imwrite(f"rectified_dvs_{save_idx:03d}.png", rectified_dvs)
                cv2.imwrite(f"rectified_ir_{save_idx:03d}.png", rectified_ir)
                cv2.imwrite(f"rectified_combined_{save_idx:03d}.png", combined)
                print(f"Saved rectified_dvs_{save_idx:03d}.png and rectified_ir_{save_idx:03d}.png")
                save_idx += 1

    finally:
        camera.stop()
        pipe.stop()
        cv2.destroyAllWindows()
        print("Clean exit.")


if __name__ == "__main__":
    run()
