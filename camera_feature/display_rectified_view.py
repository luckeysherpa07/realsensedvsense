import cv2
import numpy as np
import pyrealsense2 as rs
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

# ------------------------- Load stereo calibration -------------------------
fs = cv2.FileStorage("stereo_calibration.yaml", cv2.FILE_STORAGE_READ)

K_event = fs.getNode("K_event").mat()
dist_event = fs.getNode("dist_event").mat()
K_ir = fs.getNode("K_ir").mat()
dist_ir = fs.getNode("dist_ir").mat()
R1 = fs.getNode("R1").mat()
R2 = fs.getNode("R2").mat()
P1 = fs.getNode("P1").mat()
P2 = fs.getNode("P2").mat()
fs.release()

# ------------------------- Initialize cameras -------------------------
event_cam = Camera.from_first_available()
width, height = event_cam.width(), event_cam.height()
slice_condition = SliceCondition.make_n_us(10000)
slicer = CameraStreamSlicer(event_cam.move(), slice_condition)

pipe = rs.pipeline()
cfg = rs.config()
ir_width, ir_height = 640, 480
cfg.enable_stream(rs.stream.infrared, 1, ir_width, ir_height, rs.format.y8, 30)
pipe.start(cfg)

# ------------------------- Precompute rectification maps -------------------------
map1_e, map2_e = cv2.initUndistortRectifyMap(
    K_event, dist_event, R1, P1, (width, height), cv2.CV_32FC1
)
map1_ir, map2_ir = cv2.initUndistortRectifyMap(
    K_ir, dist_ir, R2, P2, (width, height), cv2.CV_32FC1
)  # Resize IR to match event resolution

# ------------------------- Prepare accumulation for event intensity -------------------------
accum_frame = np.zeros((height, width), dtype=np.uint16)
temp_frame = np.zeros((height, width, 3), dtype=np.uint8)
SLICES_PER_FRAME = 2
slice_counter = 0

# ------------------------- Live loop -------------------------
slicer_iter = iter(slicer)

try:
    while True:
        # ---- Event slice ----
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

        slice_counter = 0
        frame_event = accum_frame.copy()
        accum_frame[:] = 0

        # Normalize and blur event frame
        maxval = frame_event.max()
        if maxval > 0:
            frame_event = cv2.convertScaleAbs(frame_event, alpha=255.0 / maxval)
        blur_frame = cv2.GaussianBlur(frame_event, (5, 5), 0)

        # ---- Grab IR frame ----
        frames = pipe.poll_for_frames()
        if not frames:
            continue

        ir_frame = frames.get_infrared_frame(1)
        if not ir_frame:
            continue

        gray_ir = np.asanyarray(ir_frame.get_data())
        # Resize IR to match event resolution
        gray_ir_resized = cv2.resize(gray_ir, (width, height), interpolation=cv2.INTER_AREA)

        # ---- Rectify ----
        rect_event = cv2.remap(blur_frame, map1_e, map2_e, cv2.INTER_LINEAR)
        rect_ir = cv2.remap(gray_ir_resized, map1_ir, map2_ir, cv2.INTER_LINEAR)

        # ---- Side-by-side display ----
        rect_event_small = cv2.resize(rect_event, (320, 240))
        rect_ir_small = cv2.resize(rect_ir, (320, 240))
        side_by_side = np.hstack([rect_event_small, rect_ir_small])
        cv2.imshow("Rectified Event | Rectified IR", side_by_side)

        # ---- Overlay display ----
        ir_color = cv2.cvtColor(rect_ir, cv2.COLOR_GRAY2BGR)
        event_color = cv2.applyColorMap(rect_event, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(ir_color, 0.6, event_color, 0.4, 0)
        cv2.imshow("Overlay Event on IR", overlay)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
