from picamera2 import Picamera2
import time
import cv2 # Still needed for saving images, but not for display

def main():
    print("Initializing Picamera2...")
    try:
        picam2 = Picamera2(0) # Use camera 0
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        return

    print("Configuring camera...")
    try:
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
    time.sleep(2)
    
    output_folder = "captured_frames"
    try:
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Frames will be saved in '{output_folder}' directory.")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        # Continue without saving if directory creation fails, but log it.
        pass

    num_frames_to_capture = 5
    print(f"Attempting to capture {num_frames_to_capture} frames...")

    try:
        for i in range(num_frames_to_capture):
            print(f"Capturing frame {i+1}/{num_frames_to_capture}...")
            try:
                frame_rgb = picam2.capture_array()
            except Exception as e:
                print(f"Error in capture_array(): {e}")
                break # Exit loop on capture error
            
            if frame_rgb is None:
                print("Frame captured is None. Skipping save.")
                continue

            print(f"Frame {i+1} captured. Shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
            
            # Convert RGB to BGR for saving with OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            file_path = f"{output_folder}/frame_{i:02d}.jpg"
            try:
                cv2.imwrite(file_path, frame_bgr)
                print(f"Frame {i+1} saved to {file_path}")
            except Exception as e:
                print(f"Error saving frame {i+1} to {file_path}: {e}")
            
            time.sleep(0.5) # Small delay between captures
                
    except Exception as e:
        print(f"An error occurred during frame capture/saving: {e}")
    finally:
        print("Stopping camera...")
        picam2.stop()
        print("Script finished.")

if __name__ == "__main__":
    main() 