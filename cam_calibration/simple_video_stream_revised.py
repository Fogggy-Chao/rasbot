import cv2
import time

def main():
    window_name = "Live Stream"
    print(f"Attempting to create named window: '{window_name}' first...")
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create window before Picamera2 init
        # Optionally, display a placeholder or blank image briefly
        # import numpy as np
        # blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # cv2.imshow(window_name, blank_frame)
        # cv2.waitKey(100) # Show it for a very short time
        print("cv2.namedWindow() successful.")
    except Exception as e:
        print(f"Error during initial cv2.namedWindow(): {e}")
        return

    print("Initializing Picamera2...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2(0) # Use camera 0
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        cv2.destroyAllWindows()
        return

    print("Configuring Picamera2...")
    try:
        # Configure for RGB888, capture_array will return 3-channel RGB
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
    except Exception as e:
        print(f"Error configuring Picamera2: {e}")
        picam2.close() # Ensure picam2 is closed if it was initialized
        cv2.destroyAllWindows()
        return

    print("Starting Picamera2...")
    try:
        picam2.start()
    except Exception as e:
        print(f"Error starting Picamera2: {e}")
        picam2.close()
        cv2.destroyAllWindows()
        return
    
    print("Allowing camera to warm up (2 seconds)...")
    time.sleep(2) 
    
    print("Starting video stream loop... Press 'q' in the window to quit.")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            # print(f"Loop {frame_count}: Capturing frame...")
            try:
                frame_rgb = picam2.capture_array()
            except Exception as e:
                print(f"Error in capture_array(): {e}")
                break
            
            if frame_rgb is None:
                print("Frame captured is None. Skipping.")
                time.sleep(0.01) # Avoid tight loop on None frames
                continue

            # print("Converting frame to BGR...")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # print("Displaying frame...")
            cv2.imshow(window_name, frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' key pressed. Quitting...")
                break
            
            if frame_count > 1000: # Safety break after ~30 seconds at 30fps
                print("Reached 1000 frames, exiting loop as a safety measure.")
                break
                
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        print("Stopping Picamera2...")
        picam2.stop()
        # picam2.close() # picam2.stop() should be enough, close() is more for releasing the camera object itself
        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()
        print("Stream stopped and resources released.")

if __name__ == "__main__":
    main() 