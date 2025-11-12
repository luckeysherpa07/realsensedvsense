import numpy as np
import cv2
import torch
from dvsense_driver.camera_manager import DvsCameraManager


def run():
    # Initialize camera manager
    dvs_camera_manager = DvsCameraManager()
    dvs_camera_manager.update_cameras()

    # Get camera descriptions
    camera_descriptions = dvs_camera_manager.get_camera_descs()
    print(camera_descriptions)

    # Check if any cameras are available
    if not camera_descriptions:
        print("No camera found. Exiting...")
        exit(0)

    # Print all camera description information
    for camera_desc in camera_descriptions:
        print(camera_desc)

    # Open the first available camera
    try:
        camera = dvs_camera_manager.open_camera(camera_descriptions[0].serial)
    except Exception as e:
        print(f"Failed to open camera: {e}")
        exit(1)

    # Print camera information
    print(camera)

    # Get camera width and height
    width = camera.get_width()
    height = camera.get_height()

    # Define color coding dictionary
    COLOR_CODING: dict = {
        'blue_red': {
            'on': [0, 0, 255],
            'off': [255, 0, 0],
            'bg': [0, 0, 0]
        },
        'blue_white': {
            'on': [216, 223, 236],
            'off': [201, 126, 64],
            'bg': [0, 0, 0]
        }
    }

    # Start the camera and set batch event duration
    camera.start()
    camera.set_batch_events_time(10000)  # Set batch duration to 10 milliseconds

    # Display event stream in real time
    while True:
        # Get event data
        events = camera.get_next_batch()

        # Initialize data
        histogram = torch.zeros((2, height, width), dtype=torch.long)

        # Extract event x, y coordinates and polarity
        x_coords: torch.Tensor = torch.tensor(events['x'].astype(np.int32), dtype=torch.long)
        y_coords: torch.Tensor = torch.tensor(events['y'].astype(np.int32), dtype=torch.long)
        polarities: torch.Tensor = torch.tensor(events['polarity'].astype(np.int32), dtype=torch.long)

        # Update data
        torch.index_put_(
            histogram, (polarities, y_coords, x_coords), torch.ones_like(x_coords), accumulate=False
        )
        _, hist_height, hist_width = histogram.shape

        # Define color coding
        color_coding: dict = COLOR_CODING['blue_white']

        # Initialize canvas
        canvas = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        canvas[:, :] = color_coding['bg']

        # Convert data to NumPy arrays
        off_histogram, on_histogram = histogram.cpu().numpy()

        # Update canvas color based on event polarity
        canvas[on_histogram > 0] = color_coding['on']
        canvas[off_histogram > 0] = color_coding['off']

        # Display event image
        cv2.imshow('events', canvas)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()
    camera.stop()


if __name__ == "__main__":
    run()
