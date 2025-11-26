import cv2
import numpy as np
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import pyrealsense2 as rs


def run():
    # ------------------------- CONFIG -------------------------
    GRID_ROWS = 4
    GRID_COLS = 5
    BOARD_SIZE = (GRID_COLS, GRID_ROWS)
    SPACING = 3.7  # cm

    SLICES_PER_FRAME = 2
    TOTAL_VIEWS = 20
    DS = 0.5  # Downscale detection factor

    # ------------------------- Initialize Event Camera -------------------------
    event_cam = Camera.from_first_available()
    width, height = event_cam.width(), event_cam.height()
    print(f"Event camera size: {width} x {height}")

    slice_condition = SliceCondition.make_n_us(10000)
    slicer = CameraStreamSlicer(event_cam.move(), slice_condition)

    accum_frame = np.zeros((height, width), dtype=np.uint16)
    temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
    slice_counter = 0

    # Blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000
    detector = cv2.SimpleBlobDetector_create(params)

    # ------------------------- Initialize IR Camera -------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    ir_width, ir_height = 640, 480
    cfg.enable_stream(rs.stream.infrared, 1, ir_width, ir_height, rs.format.y8, 30)
    pipe.start(cfg)

    # ------------------------- Prepare Object Points -------------------------
    objp = np.zeros((GRID_ROWS * GRID_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2)
    objp *= SPACING

    object_points = []
    img_points_event = []
    img_points_ir = []

    collected_views = 0
    print(f"Collecting {TOTAL_VIEWS} synchronized views... Move the grid slowly.")

    slicer_iter = iter(slicer)

    try:
        while collected_views < TOTAL_VIEWS:

            # ------------------------- Event slice -------------------------
            try:
                slice = next(slicer_iter)
            except StopIteration:
                break

            if slice.events.size < 50:
                continue

            temp_frame[:] = 0
            BaseFrameGenerationAlgorithm.generate_frame(slice.events, temp_frame)
            gray_event = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

            accum_frame += gray_event.astype(np.uint16)
            slice_counter += 1
            if slice_counter < SLICES_PER_FRAME:
                continue

            slice_counter = 0
            process_frame = accum_frame.copy()
            accum_frame[:] = 0

            maxval = process_frame.max()
            if maxval <= 1:
                continue

            norm_frame = cv2.convertScaleAbs(
                process_frame,
                alpha=255.0 / (maxval + 1e-9)
            )
            blur_frame = cv2.GaussianBlur(norm_frame, (5, 5), 0)

            # Downscale for detection
            small_event = cv2.resize(blur_frame, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
            ret_event, centers_event_small = cv2.findCirclesGrid(
                small_event, BOARD_SIZE,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=detector
            )

            # ------------------------- IR frame -------------------------
            frames = pipe.poll_for_frames()
            if not frames:
                continue

            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue

            ir_img = np.asanyarray(ir_frame.get_data())
            ir_resized = cv2.resize(ir_img, (width, height), interpolation=cv2.INTER_AREA)

            small_ir = cv2.resize(ir_resized, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
            ret_ir, centers_ir_small = cv2.findCirclesGrid(
                small_ir, BOARD_SIZE,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=detector
            )

            # ------------------------- SYNCHRONIZED CAPTURE -------------------------
            if ret_event and ret_ir:

                collected_views += 1
                print(f"Captured synchronized view {collected_views}/{TOTAL_VIEWS}")

                centers_event = (centers_event_small / DS).astype(np.float32).reshape(-1, 1, 2)
                centers_ir = (centers_ir_small / DS).astype(np.float32).reshape(-1, 1, 2)

                # Subpixel refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
                cv2.cornerSubPix(blur_frame, centers_event, (5, 5), (-1, -1), criteria)
                cv2.cornerSubPix(ir_resized, centers_ir, (5, 5), (-1, -1), criteria)

                object_points.append(objp.copy())
                img_points_event.append(centers_event)
                img_points_ir.append(centers_ir)

                # Save images ONLY
                evt_disp = cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR)
                ir_disp = cv2.cvtColor(ir_resized, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(evt_disp, BOARD_SIZE, centers_event, True)
                cv2.drawChessboardCorners(ir_disp, BOARD_SIZE, centers_ir, True)

                cv2.imwrite(f"event_view_{collected_views}.png", evt_disp)
                cv2.imwrite(f"ir_view_{collected_views}.png", ir_disp)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    if collected_views < TOTAL_VIEWS:
        print("Not enough synchronized views. Calibration aborted.")
        return

    # ------------------------- Single-camera calibrations -------------------------
    print("\nCalibrating event camera...")
    _, K_event, dist_event, _, _ = cv2.calibrateCamera(
        object_points, img_points_event, (width, height), None, None
    )

    print("Calibrating IR camera...")
    _, K_ir, dist_ir, _, _ = cv2.calibrateCamera(
        object_points, img_points_ir, (width, height), None, None
    )

    # ------------------------- Stereo calibration -------------------------
    print("\nRunning stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        object_points, img_points_event, img_points_ir,
        K_event, dist_event, K_ir, dist_ir,
        (width, height),
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )

    # ------------------------- Stereo rectification -------------------------
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_event, dist_event, K_ir, dist_ir,
        (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # ------------------------- YAML save -------------------------
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

    print("Done. Saved to stereo_calibration.yaml")


if __name__ == "__main__":
    run()
