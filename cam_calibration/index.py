#!/usr/bin/env python3
import time, cv2
from picamera2 import Picamera2

# Setup two Picamera2 instances
cam0 = Picamera2(camera_num=0)
cam1 = Picamera2(camera_num=1)
cfg0 = cam0.create_still_configuration()
cfg1 = cam1.create_still_configuration()

cam0.configure(cfg0)
cam1.configure(cfg1)
cam0.start()
cam1.start()

for i in range(20):
    time.sleep(3)  # reposition checkerboard
    img0 = cam0.capture_array()  # NumPy array
    img1 = cam1.capture_array()

    # --- Debug prints added ---
    print(f"Loop {i}: Captured img0 shape: {img0.shape if img0 is not None else 'None'}")
    print(f"Loop {i}: Captured img1 shape: {img1.shape if img1 is not None else 'None'}")
    # --- End debug prints ---

    # --- Save with status check ---
    if img0 is not None:
        success0 = cv2.imwrite(f"./images/l/left_{i:02d}.jpg", img0)
        print(f"Loop {i}: Saving img0 to images/l/left_{i:02d}.jpg - Success: {success0}")
    else:
        print(f"Loop {i}: Skipping save for img0 (capture failed)")

    if img1 is not None:
        success1 = cv2.imwrite(f"./images/r/right_{i:02d}.jpg", img1)
        print(f"Loop {i}: Saving img1 to images/r/right_{i:02d}.jpg - Success: {success1}")
    else:
        print(f"Loop {i}: Skipping save for img1 (capture failed)")
    # --- End save with status check ---

cam0.close()
cam1.close()