import numpy as np
import cv2
import os
from dvsense_driver.camera_manager import DvsCameraManager
import pyrealsense2 as rs

def get_next_filename(folder, prefix, ext):
    os.makedirs(folder, exist_ok=True)
    existing_files = [f for f in os.listdir(folder) if f.endswith(ext)]
    numbers = []
    for f in existing_files:
        try:
            num = int(f[:4])
            numbers.append(num)
        except:
            continue
    next_num = max(numbers)+1 if numbers else 1
    return os.path.join(folder, f"{next_num:04d}_{prefix}{ext}")

def run():
    # -------------------------------
    # Initialize DVS camera
    # -------------------------------
    dvs_manager = DvsCameraManager()
    dvs_manager.update_cameras()
    cameras = dvs_manager.get_camera_descs()
    if not cameras:
        print("No DVS camera found.")
        return

    dvs_camera = dvs_manager.open_camera(cameras[0].serial)
    dvs_w = dvs_camera.get_width()
    dvs_h = dvs_camera.get_height()
    dvs_camera.start()
    dvs_camera.set_batch_events_time(10000)

    COLOR = {'on':[216,223,236], 'off':[201,126,64], 'bg':[0,0,0]}
    dvs_recording = False
    dataset_folder = "dataset"

    # -------------------------------
    # Initialize RealSense pipeline
    # -------------------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    pipe.start(cfg)

    rs_recording = False
    rgb_writer = None
    depth_writer = None
    ir_writer = None

    print("\nControls:")
    print(" SPACE = Start/Stop recording DVS + RealSense")
    print(" Q = Quit\n")

    try:
        while True:
            # -------------------------------
            # DVS events
            # -------------------------------
            events = dvs_camera.get_next_batch()
            canvas = np.zeros((dvs_h, dvs_w,3), dtype=np.uint8)
            canvas[:,:] = COLOR['bg']

            if events is not None:
                x = events['x'].astype(np.int32)
                y = events['y'].astype(np.int32)
                p = events['polarity'].astype(np.int32)
                valid = (x>=0) & (x<dvs_w) & (y>=0) & (y<dvs_h)
                x, y, p = x[valid], y[valid], p[valid]

                canvas[y[p==1], x[p==1]] = COLOR['on']
                canvas[y[p==0], x[p==0]] = COLOR['off']

            if dvs_recording:
                cv2.putText(canvas, "REC", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

            cv2.imshow("DVS Events", canvas)

            # -------------------------------
            # RealSense streams
            # -------------------------------
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            if not depth_frame or not color_frame or not ir_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow("RGB", color_image)
            cv2.imshow("Depth", depth_colormap)
            cv2.imshow("IR", ir_image)

            # Write frames if recording
            if rs_recording:
                if rgb_writer and depth_writer and ir_writer:
                    rgb_writer.write(color_image)
                    depth_writer.write(depth_colormap)
                    ir_writer.write(ir_image)

            # -------------------------------
            # Key handling
            # -------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                dvs_recording = not dvs_recording
                rs_recording = dvs_recording

                if dvs_recording:
                    # DVS recording
                    current_dvs_filename = get_next_filename(dataset_folder, "event", ".aedat4")
                    dvs_camera.start_recording(current_dvs_filename)
                    print(f"DVS recording started → {current_dvs_filename}")

                    # RealSense recording
                    current_rgb_file = get_next_filename(dataset_folder, "rgb", ".avi")
                    current_depth_file = get_next_filename(dataset_folder, "depth", ".avi")
                    current_ir_file = get_next_filename(dataset_folder, "ir", ".avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    rgb_writer = cv2.VideoWriter(current_rgb_file, fourcc, 30, (640,480))
                    depth_writer = cv2.VideoWriter(current_depth_file, fourcc, 30, (640,480))
                    ir_writer = cv2.VideoWriter(current_ir_file, fourcc, 30, (640,480))
                    print(f"RealSense recording started → {current_rgb_file}, {current_depth_file}, {current_ir_file}")

                else:
                    # Stop DVS recording
                    dvs_camera.stop_recording()
                    current_dvs_filename = None

                    # Stop RealSense recording
                    if rgb_writer: rgb_writer.release()
                    if depth_writer: depth_writer.release()
                    if ir_writer: ir_writer.release()
                    rgb_writer = depth_writer = ir_writer = None
                    print("DVS + RealSense recording stopped.")

            if key == ord('q'):
                break

    finally:
        # -------------------------------
        # Cleanup
        # -------------------------------
        if dvs_recording:
            dvs_camera.stop_recording()
        dvs_camera.stop()

        if rgb_writer: rgb_writer.release()
        if depth_writer: depth_writer.release()
        if ir_writer: ir_writer.release()
        pipe.stop()
        cv2.destroyAllWindows()
        print("All streams stopped. Exit.")

if __name__ == "__main__":
    run()
