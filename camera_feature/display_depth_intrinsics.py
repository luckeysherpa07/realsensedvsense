import pyrealsense2 as rs
import numpy as np
import cv2

def run():
    # RealSense RGB setup
    pipe = rs.pipeline()
    cfg = rs.config()
    width, height = 640, 480
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipe.start(cfg)

    # Calibration pattern
    grid_rows, grid_cols = 4, 5
    spacing = 0.05
    board_size = (grid_cols, grid_rows)

    objp = np.zeros((grid_rows * grid_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_cols, 0:grid_rows].T.reshape(-1, 2)
    objp *= spacing

    object_points = []
    img_points = []

    collected = 0
    needed = 20

    prev_gray = None
    detection_enabled = False

    MOTION_HIGH = 800_000
    MOTION_LOW  = 600_000

    print("Move the 5x4 circle grid in front of the RGB camera...")

    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        # Compute motion
        diff = cv2.absdiff(gray, prev_gray)
        motion = np.sum(diff)

        # Start detection when motion is high
        if not detection_enabled and motion > MOTION_HIGH:
            detection_enabled = True
            print(">>> Motion detected → ready to capture once stabilized")

        # Capture once motion drops below threshold
        if detection_enabled and motion < MOTION_LOW:
            ret, centers = cv2.findCirclesGrid(
                gray, board_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID
            )
            if ret:
                collected += 1
                print(f"Captured view {collected}/{needed}")
                img_points.append(centers)
                object_points.append(objp)

                display = color_img.copy()
                cv2.drawChessboardCorners(display, board_size, centers, ret)
                cv2.imwrite(f"rgb_view_{collected}.png", display)
                cv2.imshow("RGB Calibration View", display)
                cv2.waitKey(500)  # short pause to see the captured frame

                detection_enabled = False  # wait for next motion

        prev_gray = gray
        cv2.imshow("RGB Calibration View", color_img)

        if cv2.waitKey(1) == ord('q'):
            break
        if collected >= needed:
            break

    pipe.stop()
    cv2.destroyAllWindows()

    # Calibrate
    print("\n=== Calibrating RGB camera ===")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, img_points, (width, height), None, None
    )
    print("\n=== INTRINSIC MATRIX (K) ===")
    print(K)
    print("\n=== DISTORTION COEFFICIENTS ===")
    print(dist)
    print("\nCalibration complete!")

if __name__ == "__main__":
    run()
