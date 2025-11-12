from camera_feature import display_depth_1_live

def main():
    while True:
        print("\nChoose an option:")
        print("1. Display Depth 1 Live")
        print("0. Exit")
        choice = input("Enter the number: ").strip()
        if choice == "1":
            print("\nRunning Display Depth 1 Live...\n")
            display_depth_1_live.run()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()