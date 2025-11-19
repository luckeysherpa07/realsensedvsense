import cv2
import numpy as np
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

def run():
    # -------------------------
    # Open the first available camera
    # -------------------------
    camera = Camera.from_first_available()
    width, height = camera.width(), camera.height()
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
    board_size = (grid_cols, grid_rows)

    # Prepare object points (3D)
    objp = np.zeros((grid_rows * grid_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_cols, 0:grid_rows].T.reshape(-1, 2)
    spacing = 0.05  # 50 mm
    objp *= spacing

    object_points = []
    img_points = []

    # -------------------------
    # Blob detector for circle detection
    # -------------------------
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000
    detector = cv2.SimpleBlobDetector_create(params)

    # -------------------------
    # Accumulation variables
    # -------------------------
    accum_frame = np.zeros((height, width), dtype=np.uint16)
    temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
    slices_per_frame = 5
    slice_counter = 0

    total_views = 20
    collected_views = 0

    print(f"Collecting {total_views} circle grid views. Move the grid slowly...")

    for slice in slicer:
        if slice.events.size < 50:
            continue

        # Generate frame from events
        temp_frame[:] = 0
        BaseFrameGenerationAlgorithm.generate_frame(slice.events, temp_frame)
        gray_slice = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

        # Accumulate slices
        accum_frame += gray_slice.astype(np.uint16)
        slice_counter += 1

        if slice_counter >= slices_per_frame:
            slice_counter = 0
            process_frame = accum_frame.copy()
            accum_frame[:] = 0

            # Normalize and blur for better circle detection
            norm_frame = cv2.convertScaleAbs(process_frame, alpha=255.0/(process_frame.max()+1))
            blur_frame = cv2.GaussianBlur(norm_frame, (5,5), 0)

            # Detect circle grid
            ret, centers = cv2.findCirclesGrid(
                blur_frame, board_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=detector
            )

            if ret:
                collected_views += 1
                print(f"Detected view {collected_views}/{total_views}")

                # Save detected points
                img_points.append(centers.astype(np.float32))
                object_points.append(objp)

                # Draw and save detection
                vis_frame = cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis_frame, board_size, centers, ret)
                cv2.putText(vis_frame, f"View {collected_views}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imwrite(f"detected_view_{collected_views}.png", vis_frame)

                cv2.waitKey(500)  # brief pause to show detection

                if collected_views >= total_views:
                    break
            else:
                cv2.imshow("Accumulated Frame", blur_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    cv2.destroyAllWindows()

    # -------------------------
    # Camera calibration
    # -------------------------
    if len(object_points) == 0:
        print("No valid views captured. Calibration failed.")
        return

    print("Calibrating camera...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, img_points, (width, height), None, None
    )

    print("\n==== INTRINSIC MATRIX K ====")
    print(K)
    print("\n==== DISTORTION COEFFICIENTS ====")
    print(dist)
    print("Calibration complete!")

if __name__ == "__main__":
    run()
