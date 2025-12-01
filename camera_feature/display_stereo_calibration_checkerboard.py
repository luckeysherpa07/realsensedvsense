import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager

def run():
    # ---------------- Calibration parameters ----------------
    CHECKERBOARD_ROWS = 7
    CHECKERBOARD_COLS = 10
    BOARD_SIZE = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
    SQUARE_SIZE = 2.25  # cm
    NUM_IMAGES = 20
    EVENT_BATCH_TIME = 10000  # 10 ms

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    COLOR_CODING = {
        'blue_white': {
            'on': [216, 223, 236],
            'off': [201, 126, 64],
            'bg': [0, 0, 0]
        }
    }

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_dvs = []
    imgpoints_ir = []

    # ---------------- Initialize cameras ----------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipe.start(cfg)

    dvs_manager = DvsCameraManager()
    dvs_manager.update_cameras()
    camera_descs = dvs_manager.get_camera_descs()
    if not camera_descs:
        print("No DVS camera found. Exiting...")
        pipe.stop()
        return

    camera = dvs_manager.open_camera(camera_descs[0].serial)
    width, height = camera.get_width(), camera.get_height()
    camera.start()
    camera.set_batch_events_time(EVENT_BATCH_TIME)
    color_coding = COLOR_CODING['blue_white']

    print("Starting checkerboard capture... Press 'q' to quit early.")

    captured_pairs = 0
    try:
        while captured_pairs < NUM_IMAGES:
            frames = pipe.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue
            ir_image = np.asanyarray(ir_frame.get_data())

            events = camera.get_next_batch()
            if events['x'].size == 0:
                continue

            histogram = torch.zeros((2, height, width), dtype=torch.long)
            x_coords = torch.tensor(events['x'].astype(np.int32), dtype=torch.long)
            y_coords = torch.tensor(events['y'].astype(np.int32), dtype=torch.long)
            polarities = torch.tensor(events['polarity'].astype(np.int32), dtype=torch.long)
            torch.index_put_(histogram, (polarities, y_coords, x_coords), torch.ones_like(x_coords), accumulate=True)

            off_hist, on_hist = histogram.cpu().numpy()
            dvs_frame = np.zeros((height, width, 3), dtype=np.uint8)
            dvs_frame[:, :] = color_coding['bg']
            dvs_frame[on_hist > 0] = color_coding['on']
            dvs_frame[off_hist > 0] = color_coding['off']

            dvs_gray = cv2.cvtColor(dvs_frame, cv2.COLOR_BGR2GRAY)
            dvs_blur = cv2.GaussianBlur(dvs_gray, (5, 5), 0)

            display_dvs = cv2.cvtColor(dvs_blur, cv2.COLOR_GRAY2BGR)
            display_ir = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            # ---------------- Detect checkerboard ----------------
            ret_ir, corners_ir = cv2.findChessboardCorners(ir_image, BOARD_SIZE,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret_dvs, corners_dvs = cv2.findChessboardCorners(dvs_blur, BOARD_SIZE,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret_ir:
                cv2.drawChessboardCorners(display_ir, BOARD_SIZE, corners_ir, ret_ir)
            if ret_dvs:
                cv2.drawChessboardCorners(display_dvs, BOARD_SIZE, corners_dvs, ret_dvs)

            cv2.imshow("IR", display_ir)
            cv2.imshow("DVS", display_dvs)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if ret_ir and ret_dvs:
                cv2.imwrite(f"ir_frame_{captured_pairs}.png", display_ir)
                cv2.imwrite(f"dvs_frame_{captured_pairs}.png", display_dvs)
                objpoints.append(objp)
                imgpoints_ir.append(corners_ir)
                imgpoints_dvs.append(corners_dvs)
                captured_pairs += 1
                print(f"Captured pair {captured_pairs}/{NUM_IMAGES}")

    finally:
        pipe.stop()
        camera.stop()
        cv2.destroyAllWindows()

    if len(objpoints) < 2:
        print("Not enough pairs captured. Exiting...")
        return

    # ---------------- Calibrate individual cameras ----------------
    ret_ir, mtx_ir, dist_ir, rvecs_ir, tvecs_ir = cv2.calibrateCamera(
        objpoints, imgpoints_ir, ir_image.shape[::-1], None, None
    )
    print("INDIVIDUAL IR CAMERA CALIBRATION", ret_ir, mtx_ir, dist_ir)

    ret_dvs, mtx_dvs, dist_dvs, rvecs_dvs, tvecs_dvs = cv2.calibrateCamera(
        objpoints, imgpoints_dvs, (width, height), None, None
    )
    print("INDIVIDUAL EVENT CAMERA CALIBRATION", ret_dvs, mtx_dvs, dist_dvs)

    # ---------------- Stereo calibration ----------------
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx_dvs, dist_dvs, mtx_ir, dist_ir, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_dvs,
        imgpoints_ir,
        mtx_dvs, dist_dvs,
        mtx_ir, dist_ir,
        ir_image.shape[::-1],
        criteria=criteria,
        flags=flags
    )

    print("\nCalibration complete!")
    print("Rotation between cameras:\n", R)
    print("Translation between cameras:\n", T)

    # ---------------- Save to YAML ----------------
    fs = cv2.FileStorage("stereo_calibration_checkerboard.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("DVS_intrinsics", mtx_dvs)
    fs.write("DVS_distortion", dist_dvs)
    fs.write("IR_intrinsics", mtx_ir)
    fs.write("IR_distortion", dist_ir)
    fs.write("Rotation", R)
    fs.write("Translation", T)
    fs.write("Essential", E)
    fs.write("Fundamental", F)
    fs.release()
    print("Calibration saved to stereo_calibration_checkerboard.yaml")


if __name__ == "__main__":
    run()
