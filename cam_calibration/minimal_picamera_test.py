import time
from picamera2 import Picamera2

print("Attempting to initialize Picamera2...")
try:
    picam2 = Picamera2(camera_num=0)
    print("Picamera2 object created.")
    
    # Try a very simple configuration
    config = picam2.create_preview_configuration()
    print(f"Preview configuration created: {config}")
    
    picam2.configure(config)
    print("Camera configured.")
    
    picam2.start()
    print("Camera started.")
    
    time.sleep(1) # Let camera warm up
    
    # Capture an array (optional, but good test)
    # array = picam2.capture_array()
    # print(f"Frame captured, shape: {array.shape}")
    
    picam2.stop()
    print("Camera stopped.")
    picam2.close()
    print("Camera closed.")
    print("Minimal Picamera2 test successful!")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 