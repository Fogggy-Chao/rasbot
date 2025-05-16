import cv2
from picamera2 import Picamera2
import time

def main():
    print("Initializing Picamera2...")
    try:
        # Initialize Picamera2 for the first camera (index 0)
        picam2 = Picamera2(0)
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        return

    print("Configuring camera...")
    try:
        # Configure for RGB888 format, which capture_array will return as 3-channel RGB
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
    except Exception as e:
        print(f"Error configuring camera: {e}")
        return

    print("Starting camera...")
    try:
        picam2.start()
    except Exception as e:
        print(f"Error starting camera: {e}")
        return
    
    print("Allowing camera to warm up...")
    time.sleep(2) # Increased warmup time slightly
    
    print("Starting video stream loop... Press 'q' in the window to quit.")
    
    window_name = "Live Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create window beforehand

    try:
        frame_count = 0
        while True:
            frame_count += 1
            print(f"Loop iteration {frame_count}: Attempting to capture frame...")
            try:
                # Capture a frame (should be RGB)
                frame_rgb = picam2.capture_array()
            except Exception as e:
                print(f"Error in capture_array(): {e}")
                break
            
            if frame_rgb is None:
                print("Frame captured is None. Skipping.")
                continue

            print(f"Frame captured. Shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
            
            # Convert RGB to BGR for OpenCV
            print("Converting frame to BGR...")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            print("Attempting to display frame...")
            try:
                cv2.imshow(window_name, frame_bgr)
            except Exception as e:
                print(f"Error in cv2.imshow(): {e}")
                # If imshow fails, we might want to break or continue without display
                # For now, let's try to continue to see if capture still works
                # but we must have a waitKey to process OpenCV events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting due to 'q' press after imshow error.")
                    break
                continue # Skip rest of loop if imshow fails

            print("Frame display attempt finished. Waiting for key...")
            
            # Wait for a key press (1 millisecond delay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' key pressed. Quitting...")
                break
            
            if frame_count > 500: # Safety break
                print("Reached 500 frames, exiting loop.")
                break
                
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        print("Stopping camera...")
        picam2.stop()
        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()
        print("Stream stopped and resources released.")

if __name__ == "__main__":
    main() 