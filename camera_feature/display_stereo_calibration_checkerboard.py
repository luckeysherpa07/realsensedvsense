#!/usr/bin/env python3
"""
Event + RealSense IR stereo calibration using a checkerboard (findChessboardCorners).

Board provided by user:
 - inner corners: 10 x 7  (columns x rows)
 - square size: 2.25 cm

Dependencies:
 - OpenCV (cv2)
 - metavision_sdk_core, metavision_sdk_stream (for event camera)
 - pyrealsense2
 - numpy
"""

import cv2
import numpy as np
import time
import sys
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import pyrealsense2 as rs


# ------------------------- CONFIG (user-provided) -------------------------
GRID_COLS = 10              # inner corners horizontally
GRID_ROWS = 7               # inner corners vertically
BOARD_SIZE = (GRID_COLS, GRID_ROWS)  # (cols, rows) for findChessboardCorners
SQUARE_SIZE = 2.25          # cm

SLICES_PER_FRAME = 3        # accumulate this many event slices per detection (tune)
TOTAL_VIEWS = 20            # number of synchronized views to collect
DS = 0.5                    # downscale factor for detection (0.5 => half size)
MIN_EVENT_COUNT = 50        # minimal events in a slice (tune)
IR_WIDTH, IR_HEIGHT = 640, 480

# Checkerboard detection flags (fast + robust)
CB_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE |
            cv2.CALIB_CB_FAST_CHECK)


def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Compute total reprojection error for a single camera calibration result.
    Returns RMS error.
    """
    total_error = 0.0
    total_points = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)
        total_error += err * err
        total_points += len(object_points[i])
    if total_points == 0:
        return float('inf')
    return np.sqrt(total_error / total_points)


def run():
    # ------------------------- Init event camera -------------------------
    try:
        event_cam = Camera.from_first_available()
    except Exception as e:
        print("Error: couldn't open event camera:", e)
        return

    width, height = event_cam.width(), event_cam.height()
    print(f"Event camera size: {width} x {height}")

    slice_condition = SliceCondition.make_n_us(10000)  # 10 ms slices; tune if you want
    slicer = CameraStreamSlicer(event_cam.move(), slice_condition)
    slicer_iter = iter(slicer)

    accum_frame = np.zeros((height, width), dtype=np.uint16)
    temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
    slice_counter = 0

    # ------------------------- Init RealSense IR -------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, IR_WIDTH, IR_HEIGHT, rs.format.y8, 30)
    try:
        pipe.start(cfg)
    except Exception as e:
        print("Error: couldn't start RealSense pipeline:", e)
        return

    # ------------------------- Prepare object points -------------------------
    objp = np.zeros((GRID_ROWS * GRID_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # units: cm

    object_points = []   # list of (Nx3) object points
    img_points_event = []  # list of (Nx1x2) event image points
    img_points_ir = []     # list of (Nx1x2) ir image points

    collected_views = 0
    print(f"\nCollecting {TOTAL_VIEWS} synchronized views with checkerboard ({BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners)... Move board slowly.")

    try:
        while collected_views < TOTAL_VIEWS:
            # ------------------------- Get next event slice -------------------------
            try:
                slice = next(slicer_iter)
            except StopIteration:
                print("Event slicer exhausted.")
                break

            if slice is None or slice.events is None:
                continue

            if slice.events.size < MIN_EVENT_COUNT:
                continue

            temp_frame.fill(0)
            BaseFrameGenerationAlgorithm.generate_frame(slice.events, temp_frame)
            gray_event = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

            accum_frame += gray_event.astype(np.uint16)
            slice_counter += 1
            if slice_counter < SLICES_PER_FRAME:
                continue

            # process accumulated event frame
            slice_counter = 0
            process_frame = accum_frame.copy()
            accum_frame.fill(0)

            maxval = int(process_frame.max())
            if maxval <= 1:
                continue

            # Normalize and blur to create strong edges for checkerboard
            norm_frame = cv2.convertScaleAbs(process_frame, alpha=255.0 / (maxval + 1e-9))
            blur_event = cv2.GaussianBlur(norm_frame, (5, 5), 0)

            # Downscale for faster detection
            small_event = cv2.resize(blur_event, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)

            # ------------------------- Get IR frame (non-blocking) -------------------------
            frames = pipe.poll_for_frames()
            if not frames:
                continue
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue
            ir_img = np.asanyarray(ir_frame.get_data())
            # resize IR to event camera resolution for correspondence
            ir_resized = cv2.resize(ir_img, (width, height), interpolation=cv2.INTER_AREA)
            small_ir = cv2.resize(ir_resized, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)

            # ------------------------- Checkerboard detection -------------------------
            ret_event, corners_event_small = cv2.findChessboardCorners(
                small_event, BOARD_SIZE, flags=CB_FLAGS
            )

            ret_ir, corners_ir_small = cv2.findChessboardCorners(
                small_ir, BOARD_SIZE, flags=CB_FLAGS
            )

            if ret_event and ret_ir and (corners_event_small is not None) and (corners_ir_small is not None):
                # scale back to original resolution
                try:
                    corners_event = (corners_event_small / DS).astype(np.float32)
                    corners_ir = (corners_ir_small / DS).astype(np.float32)
                except Exception as e:
                    print("Warning: scaling corners failed:", e)
                    continue

                # prepare images for cornerSubPix (must be single-channel 8-bit or float32)
                event_for_subpix = np.float32(blur_event)
                ir_for_subpix = np.float32(ir_resized)

                # Criteria for corner refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0005)

                # refine
                try:
                    cv2.cornerSubPix(event_for_subpix, corners_event, (5, 5), (-1, -1), criteria)
                    cv2.cornerSubPix(ir_for_subpix, corners_ir, (5, 5), (-1, -1), criteria)
                except Exception as e:
                    print("cornerSubPix warning:", e)

                # Accept this synchronized pair
                collected_views += 1
                print(f"Captured synchronized view {collected_views}/{TOTAL_VIEWS}")

                # Append points in required formats
                object_points.append(objp.copy())
                img_points_event.append(corners_event.reshape(-1, 1, 2))
                img_points_ir.append(corners_ir.reshape(-1, 1, 2))

                # Save visualization images
                evt_disp = cv2.cvtColor(blur_event, cv2.COLOR_GRAY2BGR)
                ir_disp = cv2.cvtColor(ir_resized, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(evt_disp, BOARD_SIZE, corners_event, True)
                cv2.drawChessboardCorners(ir_disp, BOARD_SIZE, corners_ir, True)
                cv2.imwrite(f"event_view_{collected_views:02d}.png", evt_disp)
                cv2.imwrite(f"ir_view_{collected_views:02d}.png", ir_disp)

            # short sleep to free CPU
            time.sleep(0.001)

    finally:
        # Always try to stop RealSense pipeline
        try:
            pipe.stop()
        except Exception:
            pass

    if collected_views < TOTAL_VIEWS:
        print(f"\nNot enough synchronized views ({collected_views}/{TOTAL_VIEWS}). Aborting calibration.")
        return

    # ------------------------- Single-camera calibrations -------------------------
    print("\nCalibrating event camera...")
    ret_ev, K_event, dist_event, rvecs_ev, tvecs_ev = cv2.calibrateCamera(
        object_points, img_points_event, (width, height), None, None
    )
    ev_error = compute_reprojection_error(object_points, img_points_event, rvecs_ev, tvecs_ev, K_event, dist_event)
    print(f"Event camera RMS reprojection error (returned): {ret_ev:.6f}, computed RMS: {ev_error:.6f}")

    print("\nCalibrating IR camera...")
    ret_ir, K_ir, dist_ir, rvecs_ir, tvecs_ir = cv2.calibrateCamera(
        object_points, img_points_ir, (width, height), None, None
    )
    ir_error = compute_reprojection_error(object_points, img_points_ir, rvecs_ir, tvecs_ir, K_ir, dist_ir)
    print(f"IR camera RMS reprojection error (returned): {ret_ir:.6f}, computed RMS: {ir_error:.6f}")

    # ------------------------- Stereo calibration -------------------------
    print("\nRunning stereo calibration (fixing intrinsics)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    retval, K_event_out, dist_event_out, K_ir_out, dist_ir_out, R, T, E, F = cv2.stereoCalibrate(
        object_points, img_points_event, img_points_ir,
        K_event, dist_event, K_ir, dist_ir,
        (width, height),
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )
    print("stereoCalibrate retval (RMS):", retval)

    # ------------------------- Stereo rectification -------------------------
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_event, dist_event, K_ir, dist_ir,
        (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # ------------------------- Save YAML -------------------------
    print("\nSaving stereo_calibration.yaml ...")
    fs = cv2.FileStorage("stereo_calibration.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("K_event", K_event)
    fs.write("dist_event", dist_event)
    fs.write("K_ir", K_ir)
    fs.write("dist_ir", dist_ir)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.release()
    print("Done. Saved to stereo_calibration.yaml\n")

    print("Summary:")
    print(f" - Collected views: {collected_views}")
    print(f" - Event reprojection RMS (computed): {ev_error:.6f}")
    print(f" - IR reprojection RMS (computed):    {ir_error:.6f}")
    print(f" - Stereo calibration RMS:            {retval:.6f}")


if __name__ == "__main__":
    run()
