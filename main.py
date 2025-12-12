from camera_feature import display_depth_1_live
from camera_feature import display_event_live
from camera_feature import display_event_live_dvsense_drive
from camera_feature import display_depth_1_event_live
from camera_feature import display_event_intrinsics
from camera_feature import display_depth_intrinsics
from camera_feature import display_stereo_calibration
from camera_feature import display_stereo_calibration_checkerboard
from camera_feature import display_tempral_aligned_view
from camera_feature import display_rectified_view
from camera_feature import display_rectified_view_simplify
from camera_feature import display_depth_ir_difference_view
from camera_feature import record_depth_ir_rgb_event
from camera_feature import playback_recorded_files
from camera_feature import calibrate_frame_level_offset
from camera_feature import playback_spatial_aligned_file

def main():
    options = {
        "1": ("Display Depth 1 Live", display_depth_1_live.run),
        "2": ("Display Event Live", display_event_live.run),
        "3": ("Display Event Live from DVSense Driver", display_event_live_dvsense_drive.run),
        "4": ("Display Depth 1 and Event Live", display_depth_1_event_live.run),
        "5": ("Display Event Intrinsics", display_event_intrinsics.run),
        "6": ("Display Depth Intrinsics", display_depth_intrinsics.run),
        "7": ("Display Stereo Calibration", display_stereo_calibration.run),
        "8": ("Display Stereo Calibration Checkerboard", display_stereo_calibration_checkerboard.run),
        "9": ("Display Temporal Aligned View", display_tempral_aligned_view.run),
        "10": ("Display Rectified View", display_rectified_view.run),
        "11": ("Display Rectified View Simpilfy", display_rectified_view_simplify.run),
        "12": ("Display Depth IR Difference View", display_depth_ir_difference_view.run),
        "13": ("Record Depth IR RGB Event", record_depth_ir_rgb_event.run),
        "14": ("Playback Recorded Files", playback_recorded_files.run),
        "15": ("Calibrate Frame Level Offset", calibrate_frame_level_offset.run),
        "16": ("Playback Spatial Aligned Files", playback_spatial_aligned_file.run),
    }

    while True:
        print("\nChoose an option:")
        for key, (description, _) in options.items():
            print(f"{key}. {description}")
        print("0. Exit")

        choice = input("Enter the number: ").strip()
        if choice == "0":
            print("Exiting...")
            break
        elif choice in options:
            print(f"\nRunning {options[choice][0]}...\n")
            options[choice][1]()  # Call the function
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
