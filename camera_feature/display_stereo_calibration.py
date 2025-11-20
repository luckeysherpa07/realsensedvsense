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

    SLICES_PER_FRAME = 2        # << Faster
    TOTAL_VIEWS = 20

    DS = 0.5                    # Downscale factor

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

    # ------------------------- Initialize RGB Camera -------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    rgb_width, rgb_height = 640, 480
    cfg.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, 30)
    pipe.start(cfg)

    # ------------------------- Prepare Object Points -------------------------
    objp = np.zeros((GRID_ROWS * GRID_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2)
    objp *= SPACING

    object_points = []
    img_points_event = []
    img_points_rgb = []

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

        # -------------------- RGB camera poll_for_frames --------------------
        frames = pipe.poll_for_frames()
        if not frames:
            continue  # no blocking → MUCH faster

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        gray_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # -------------------- DOWNSCALE RGB FRAME --------------------
        small_rgb = cv2.resize(gray_rgb, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)

        # Circle detection in RGB
        ret_rgb, centers_rgb_small = cv2.findCirclesGrid(
            small_rgb, BOARD_SIZE,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )

        # -------------------- SYNCHRONIZED CAPTURE --------------------
        if ret_event and ret_rgb:

            collected_views += 1
            print(f"Captured synchronized view {collected_views}/{TOTAL_VIEWS}")

            # UPSCALE detected points
            centers_event = centers_event_small / DS
            centers_rgb = centers_rgb_small / DS

            object_points.append(objp)
            img_points_event.append(centers_event.astype(np.float32))
            img_points_rgb.append(centers_rgb.astype(np.float32))

            # Visualization
            disp_event = cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(disp_event, BOARD_SIZE, centers_event, True)
            cv2.imwrite(f"event_view_{collected_views}.png", disp_event)

            disp_rgb = color_img.copy()
            cv2.drawChessboardCorners(disp_rgb, BOARD_SIZE, centers_rgb, True)
            cv2.imwrite(f"rgb_view_{collected_views}.png", disp_rgb)

        # Optional displays
        cv2.imshow("Event Camera (downscaled)", small_event)
        cv2.imshow("RGB Camera (downscaled)", small_rgb)
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

    print("Calibrating RGB camera...")
    ret_rgb, K_rgb, dist_rgb, _, _ = cv2.calibrateCamera(
        object_points, img_points_rgb, (rgb_width, rgb_height), None, None
    )

    # ------------------------- Stereo calibration -------------------------
    print("\nRunning stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC

    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        object_points, img_points_event, img_points_rgb,
        K_event, dist_event, K_rgb, dist_rgb,
        (max(width, rgb_width), max(height, rgb_height)),
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )

    print("\n=== Stereo Calibration Results ===")
    print("Rotation R (event → RGB):\n", R)
    print("Translation T (event → RGB):\n", T)
    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_event, dist_event, K_rgb, dist_rgb,
        (max(width, rgb_width), max(height, rgb_height)),
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    print("\nRectification complete. Ready for stereo processing.")


if __name__ == "__main__":
    run()
