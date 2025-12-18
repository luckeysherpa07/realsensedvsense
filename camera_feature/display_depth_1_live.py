import pyrealsense2 as rs
import numpy as np
import cv2

def run():
    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # Note: Stream index 1 is usually the left IR camera
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

    # 1. Capture the profile when starting the pipeline
    profile = pipe.start(cfg)

    # 2. Get the specific depth sensor (which controls the emitter)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    # Optional: Start with emitter on (1) or off (0)
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    print("Press 'e' to toggle emitter. Press 'q' to quit.")

    while True: 
        frame = pipe.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()
        ir_frame = frame.get_infrared_frame(1) 

        if not depth_frame or not color_frame or not ir_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())
        
        # applyColorMap creates a visualization, it is not raw depth data
        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.08), cv2.COLORMAP_JET)

        cv2.imshow('rgb', color_image)
        cv2.imshow('depth', depth_cm)
        cv2.imshow('ir', ir_image)

        # 3. Handle Key Presses
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('e'):
            # Check if sensor supports the option to avoid crashing
            if depth_sensor.supports(rs.option.emitter_enabled):
                # Get current status (returns float, typically 1.0 or 0.0)
                current_value = depth_sensor.get_option(rs.option.emitter_enabled)
                
                # Toggle: If 1 set to 0, otherwise set to 1
                new_value = 0 if current_value == 1 else 1
                
                depth_sensor.set_option(rs.option.emitter_enabled, new_value)
                print(f"Emitter toggled: {'ON' if new_value else 'OFF'}")

    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()