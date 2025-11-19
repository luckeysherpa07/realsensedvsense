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

    print("Move the circle grid slowly in front of the camera...")

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
                print("Circle grid detected!")
                vis_frame = cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis_frame, board_size, centers, ret)
                cv2.imshow("Detected Circles", vis_frame)
                cv2.waitKey(0)
                break
            else:
                cv2.imshow("Accumulated Frame", blur_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
