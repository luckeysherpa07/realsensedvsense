from camera_feature import display_depth_1_live
from camera_feature import display_event_live
from camera_feature import display_depth_1_event_live
from camera_feature import display_calibration_info

def main():
    options = {
        "1": ("Display Depth 1 Live", display_depth_1_live.run),
        "2": ("Display Event Live", display_event_live.run),
        "3": ("Display Depth 1 and Event Live", display_depth_1_event_live.run),
        "4": ("Display Calibration Info", display_calibration_info.run),
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
