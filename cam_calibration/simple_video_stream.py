import cv2
from picamera2 import Picamera2
import time

def main():
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure the camera (e.g., for preview)
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    
    # Allow camera to warm up
    time.sleep(1) 
    
    print("Starting video stream... Press 'q' to quit.")
    
    try:
        while True:
            # Capture a frame
            frame = picam2.capture_array()
            
            # Convert to BGR for OpenCV if it's RGB (Picamera2 default)
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Not needed if format is already BGR
            
            # Display the frame
            cv2.imshow("Live Stream", frame)
            
            # Wait for a key press (1 millisecond delay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
    finally:
        # Stop the camera
        picam2.stop()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Stream stopped and resources released.")

if __name__ == "__main__":
    main() 