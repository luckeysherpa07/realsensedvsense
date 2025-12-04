import pyrealsense2 as rs
import numpy as np
import cv2

def run():
    # Configure RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipe.start(cfg)

    
    # Turn OFF IR emitter if supported
    device = profile.get_device()
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.emitter_enabled):
            #----------USE THIS TO CHANGE THE EMMITOR ON AND OFF------------
            sensor.set_option(rs.option.emitter_enabled, 0) 
            print("IR emitter turned OFF")

    try:
        while True:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            # Convert depth to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())

            # Colorize depth for visualization
            depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow("Depth View (IR emitter OFF)", depth_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
