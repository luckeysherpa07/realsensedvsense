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
    SPACING = 3.7  # 3.7 cm

    SLICES_PER_FRAME = 2
    TOTAL_VIEWS = 20

    DS = 0.5  # Downscale factor

    # ----------------------------------------------------------

    # ------------------------- Initialize Event Camera -------------------------
    event_cam = Camera.from_first_available()
    width, height = event_cam.width(), event_cam.height()
    print(f"Event camera size: {width} x {height}")

    slice_condition = SliceCondition.make_n_us(10000)
    slicer = CameraStreamSlicer(event_cam.move(), slice_condition)

    accum_frame = np.zeros((height, width), dtype=np.uint16)
    temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
    slice_counter = 0

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000
    detector = cv2.SimpleBlobDetector_create(params)

    # ------------------------- Initialize Infrared Camera -------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    ir_width, ir_height = 640, 480
    cfg.enable_stream(rs.stream.infrared, 1, ir_width, ir_height, rs.format.y8, 30)  # stream 1 is IR1
    pipe.start(cfg)

    # ------------------------- Prepare Object Points -------------------------
    objp = np.zeros((GRID_ROWS * GRID_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2)
    objp *= SPACING

    object_points = []
    img_points_event = []
    img_points_ir = []

    collected_views = 0
    print(f"Collecting {TOTAL_VIEWS} synchronized views... Move the grid slowly!")

    # ------------------------- Main Loop -------------------------
    slicer_iter = iter(slicer)

    while collected_views < TOTAL_VIEWS:

        # --- Event camera slice ---
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

        # Reset accumulator
        slice_counter = 0
        process_frame = accum_frame.copy()
        accum_frame[:] = 0

        # Normalize and blur
        norm_frame = cv2.convertScaleAbs(process_frame, alpha=255.0 / (process_frame.max() + 1e-6))
        blur_frame = cv2.GaussianBlur(norm_frame, (5, 5), 0)

        # -------------------- DOWNSCALE EVENT FRAME --------------------
        small_event = cv2.resize(blur_frame, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)

        # Circle detection on event
        ret_event, centers_event_small = cv2.findCirclesGrid(
            small_event, BOARD_SIZE,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=detector
        )

        # -------------------- IR camera poll_for_frames --------------------
        frames = pipe.poll_for_frames()
        if not frames:
            continue

        ir_frame = frames.get_infrared_frame(1)  # IR stream 1
        if not ir_frame:
            continue

        ir_img = np.asanyarray(ir_frame.get_data())
        # IR is already grayscale
        gray_ir = ir_img

        # -------------------- DOWNSCALE IR FRAME --------------------
        small_ir = cv2.resize(gray_ir, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)

        # Circle detection in IR
        ret_ir, centers_ir_small = cv2.findCirclesGrid(
            small_ir, BOARD_SIZE,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )

        # -------------------- SYNCHRONIZED CAPTURE --------------------
        if ret_event and ret_ir:

            collected_views += 1
            print(f"Captured synchronized view {collected_views}/{TOTAL_VIEWS}")

            # UPSCALE detected points
            centers_event = centers_event_small / DS
            centers_ir = centers_ir_small / DS

            object_points.append(objp)
            img_points_event.append(centers_event.astype(np.float32))
            img_points_ir.append(centers_ir.astype(np.float32))

            # Visualization
            disp_event = cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(disp_event, BOARD_SIZE, centers_event, True)
            cv2.imwrite(f"event_view_{collected_views}.png", disp_event)

            disp_ir = cv2.cvtColor(gray_ir, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(disp_ir, BOARD_SIZE, centers_ir, True)
            cv2.imwrite(f"ir_view_{collected_views}.png", disp_ir)

        # Optional displays
        cv2.imshow("Event Camera (downscaled)", small_event)
        cv2.imshow("IR Camera (downscaled)", small_ir)
        if cv2.waitKey(1) == ord('q'):
            break

    # ------------------------- Shutdown -------------------------
    pipe.stop()
    cv2.destroyAllWindows()

    if collected_views < TOTAL_VIEWS:
        print("Not enough synchronized views captured. Calibration aborted.")
        return

    # ------------------------- Single-camera calibration -------------------------
    print("\nCalibrating event camera...")
    ret_event, K_event, dist_event, _, _ = cv2.calibrateCamera(
        object_points, img_points_event, (width, height), None, None
    )

    print("Calibrating IR camera...")
    ret_ir, K_ir, dist_ir, _, _ = cv2.calibrateCamera(
        object_points, img_points_ir, (ir_width, ir_height), None, None
    )

    # ------------------------- Stereo calibration -------------------------
    print("\nRunning stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC

    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        object_points, img_points_event, img_points_ir,
        K_event, dist_event, K_ir, dist_ir,
        (max(width, ir_width), max(height, ir_height)),
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )

    print("\n=== Stereo Calibration Results ===")
    print("Rotation R (event → IR):\n", R)
    print("Translation T (event → IR):\n", T)
    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_event, dist_event, K_ir, dist_ir,
        (max(width, ir_width), max(height, ir_height)),
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    print("\nRectification complete. Ready for stereo processing.")


if __name__ == "__main__":
    run()
