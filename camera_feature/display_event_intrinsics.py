import cv2
import numpy as np
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition


def run():
    # -------------------------
    # Open the first available camera
    # -------------------------
    camera = Camera.from_first_available()
    width = camera.width()
    height = camera.height()
    print(f"Sensor size: {width} x {height}")

    # -------------------------
    # Slice events in 10 ms chunks
    # -------------------------
    slice_condition = SliceCondition.make_n_us(10000)  # 10 ms per slice
    slicer = CameraStreamSlicer(camera.move(), slice_condition)

    # -------------------------
    # Circle grid parameters (5x4)
    # -------------------------
    grid_rows = 4
    grid_cols = 5
    spacing = 0.05  # 50 mm
    board_size = (grid_cols, grid_rows)

    # Prepare object points (3D)
    objp = np.zeros((grid_rows * grid_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_cols, 0:grid_rows].T.reshape(-1, 2)
    objp *= spacing

    object_points = []
    img_points = []

    # -------------------------
    # Frame accumulation per motion
    # -------------------------
    accum_frame = np.zeros((height, width), dtype=np.uint8)
    temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
    slices_per_frame = 5
    slice_counter = 0

    collected_views = 0
    total_views = 20
    print("Move the 5x4 circle grid slowly in front of the camera...")

    # -------------------------
    # Main loop
    # -------------------------
    for slice in slicer:
        #Slice event slice
        if slice.events.size < 500:
            continue    

        BaseFrameGenerationAlgorithm.generate_frame(slice.events, temp_frame)
        gray_slice = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

        accum_frame = cv2.add(accum_frame, gray_slice)
        slice_counter += 1

        if slice_counter >= slices_per_frame:
            slice_counter = 0
            process_frame = accum_frame.copy()
            accum_frame[:] = 0

            # Detect circle grid
            ret, centers = cv2.findCirclesGrid(
                process_frame, board_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID
            )

            if ret:
                collected_views += 1
                print(f"Detected view {collected_views}/{total_views}")
                img_points.append(centers)
                object_points.append(objp)

                cv2.drawChessboardCorners(process_frame, board_size, centers, ret)
                cv2.putText(process_frame, "Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imwrite(f"detected_view_{collected_views}.png", process_frame)

            cv2.imshow("Calibration", cv2.convertScaleAbs(process_frame))
            if cv2.waitKey(1) == ord('q'):
                break

            if collected_views >= total_views:
                break

    # -------------------------
    # Calibration
    # -------------------------
    print("Calibrating camera...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, img_points, (width, height), None, None
    )

    print("\n==== INTRINSIC MATRIX K ====")
    print(K)
    print("\n==== DISTORTION COEFFICIENTS ====")
    print(dist)

    cv2.destroyAllWindows()
    print("Calibration complete!")


if __name__ == "__main__":
    run()
