import cv2
import numpy as np

def main():
    print("Creating a blank image...")
    # Create a blank 300x300 black image
    width, height = 300, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    window_name = "OpenCV Test Window"
    print(f"Attempting to create named window: '{window_name}'...")
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("cv2.namedWindow() successful.")
    except Exception as e:
        print(f"Error during cv2.namedWindow(): {e}")
        return

    print(f"Attempting to display image in window: '{window_name}'...")
    try:
        cv2.imshow(window_name, image)
        print("cv2.imshow() successful. A window should appear.")
    except Exception as e:
        print(f"Error during cv2.imshow(): {e}")
        cv2.destroyAllWindows() # Clean up if imshow fails after namedWindow succeeded
        return

    print("Press any key in the OpenCV window to close it...")
    try:
        key = cv2.waitKey(0) # Wait indefinitely for a key press
        print(f"Key pressed: {key}. Closing window.")
    except Exception as e:
        print(f"Error during cv2.waitKey(): {e}")
    finally:
        print("Destroying all OpenCV windows...")
        cv2.destroyAllWindows()
        print("Script finished.")

if __name__ == "__main__":
    main() 