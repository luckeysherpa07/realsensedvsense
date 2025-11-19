import pyrealsense2 as rs
import numpy as np
import cv2

def run():
    # RealSense IR setup
    pipe = rs.pipeline()
    cfg = rs.config()

    width, height = 640, 480
    cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)
    pipe.start(cfg)

    # Calibration pattern
    grid_rows, grid_cols = 4, 5
    spacing = 3.7
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

    print("Move the 5x4 circle grid in front of the **Infrared Camera**...")

    while True:
        frames = pipe.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1)
        if not ir_frame:
            continue

        # IR frame comes already in grayscale
        ir_img = np.asanyarray(ir_frame.get_data())
        gray = ir_img.copy()

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
                print(f"Captured IR view {collected}/{needed}")
                img_points.append(centers)
                object_points.append(objp)

                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(display, board_size, centers, ret)
                cv2.imwrite(f"ir_view_{collected}.png", display)
                cv2.imshow("IR Calibration View", display)
                cv2.waitKey(500)

                detection_enabled = False

        prev_gray = gray
        cv2.imshow("IR Calibration View", gray)

        if cv2.waitKey(1) == ord('q'):
            break
        if collected >= needed:
            break

    pipe.stop()
    cv2.destroyAllWindows()

    # Calibrate
    print("\n=== Calibrating **Infrared** camera ===")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, img_points, (width, height), None, None
    )

    print("\n=== IR INTRINSIC MATRIX (K) ===")
    print(K)
    print("\n=== IR DISTORTION COEFFICIENTS ===")
    print(dist)
    print("\nIR Calibration complete!")

if __name__ == "__main__":
    run()
