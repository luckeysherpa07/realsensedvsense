from camera_feature import display_depth_1_live
from camera_feature import display_event_live
from camera_feature import display_event_live_dvsense_drive
from camera_feature import display_depth_1_event_live
from camera_feature import display_event_intrinsics
from camera_feature import display_depth_intrinsics
from camera_feature import display_stereo_calibration

def main():
    options = {
        "1": ("Display Depth 1 Live", display_depth_1_live.run),
        "2": ("Display Event Live", display_event_live.run),
        "3": ("Display Event Live from DVSense Driver", display_event_live_dvsense_drive.run),
        "4": ("Display Depth 1 and Event Live", display_depth_1_event_live.run),
        "5": ("Display Event Intrinsics", display_event_intrinsics.run),
        "6": ("Display Depth Intrinsics", display_depth_intrinsics.run),
        "7": ("Display Stereo Calibration", display_stereo_calibration.run),
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
