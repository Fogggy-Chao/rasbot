import cv2
import numpy as np
import time
# from picamera2 import Picamera2 # Moved
import os
from ultralytics import YOLO # Added Ultralytics import

# --- Calibration and Model Constants ---
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MODEL_INPUT_WIDTH = 640 # This might be dictated by the model itself, Ultralytics might handle resizing
MODEL_INPUT_HEIGHT = 640

# Path to the directory of the model that Ultralytics can load
MODEL_DIR_PATH = "../yolo/yolo11n_ncnn_model/" 

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4 # May not be needed if Ultralytics handles NMS

# CLASS_NAMES will be loaded from the model by Ultralytics
# Keep TARGET_CLASSES as a list of strings
TARGET_CLASSES = ["cup", "bottle", "bowl", "apple"]

# Global variable to store class names loaded from model
LOADED_CLASS_NAMES = {}

# --- Load Calibration Parameters ---
def load_calibration_files(intrinsics_file, extrinsics_file):
    """Loads intrinsic and extrinsic calibration parameters from YAML files."""
    fs_intrinsics = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)
    fs_extrinsics = cv2.FileStorage(extrinsics_file, cv2.FILE_STORAGE_READ)

    if not fs_intrinsics.isOpened():
        print(f"Error: Could not open intrinsics file: {intrinsics_file}")
        return None
    if not fs_extrinsics.isOpened():
        print(f"Error: Could not open extrinsics file: {extrinsics_file}")
        return None

    # Intrinsics
    K1 = fs_intrinsics.getNode("M1").mat()
    D1 = fs_intrinsics.getNode("D1").mat()
    K2 = fs_intrinsics.getNode("M2").mat()
    D2 = fs_intrinsics.getNode("D2").mat()

    # Extrinsics
    R = fs_extrinsics.getNode("R").mat()
    T = fs_extrinsics.getNode("T").mat()
    R1 = fs_extrinsics.getNode("R1").mat()
    R2 = fs_extrinsics.getNode("R2").mat()
    P1 = fs_extrinsics.getNode("P1").mat()
    P2 = fs_extrinsics.getNode("P2").mat()
    Q = fs_extrinsics.getNode("Q").mat()

    fs_intrinsics.release()
    fs_extrinsics.release()

    return K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q

# --- Load YOLO Model (using Ultralytics) ---
def load_yolo_model(model_path):
    global LOADED_CLASS_NAMES
    print(f"Loading model using Ultralytics YOLO from: {model_path}")
    abs_model_path = os.path.abspath(model_path)
    print(f"  Absolute model path: {abs_model_path}")
    try:
        model = YOLO(abs_model_path, task='detect')
        LOADED_CLASS_NAMES = model.names # Load class names from the model
        print("Ultralytics YOLO model loaded successfully.")
        print(f"  Model classes: {LOADED_CLASS_NAMES}")
    except Exception as e:
        print(f"Error loading Ultralytics YOLO model: {e}")
        return None
    return model

# --- Object Detection Function (using Ultralytics) ---
def detect_objects(image, ultralytics_model, target_class_names_set, input_width, input_height, conf_threshold):
    detected_objects = []
    if ultralytics_model is None:
        return detected_objects

    # Perform inference using Ultralytics model
    # Ultralytics handles image resizing and normalization internally based on model needs.
    # verbose=False suppresses console output from the model itself during inference.
    try:
        predictions = ultralytics_model(image, verbose=False, imgsz=input_width) # Use input_width for imgsz for now
    except Exception as e:
        print(f"Error during Ultralytics model inference: {e}")
        return detected_objects

    if not predictions or not predictions[0].boxes:
        return detected_objects

    # Process detections
    for box in predictions[0].boxes:
        try:
            confidence = float(box.conf.item())
            if confidence < conf_threshold:
                continue

            class_id = int(box.cls.item())
            class_name = LOADED_CLASS_NAMES.get(class_id, "Unknown")

            if class_name not in target_class_names_set:
                continue

            # Extract coordinates in xyxy format
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            
            # Convert to xywh format for consistency with previous script structure
            width = xmax - xmin
            height = ymax - ymin
            box_xywh = [xmin, ymin, width, height]

            detected_objects.append({
                "class_name": class_name,
                "confidence": confidence,
                "box_xywh": box_xywh
            })
        except Exception as e:
            print(f"Error processing a detection box: {e}")
            continue # Skip this box and try the next one
            
    return detected_objects

# --- 3D Triangulation ---
def get_3d_coordinates(point_left, point_right, P1, P2):
    if point_left is None or point_right is None:
        print("get_3d_coordinates: point_left or point_right is None")
        return None

    # OpenCV expects 2xN arrays for projPoints1 and projPoints2 (N=1 in our case)
    pt_l = np.array([[point_left[0]], [point_left[1]]], dtype=np.float32)
    pt_r = np.array([[point_right[0]], [point_right[1]]], dtype=np.float32)

    # Print detailed debug information
    print(f"Triangulating with left point: {pt_l.flatten()}, right point: {pt_r.flatten()}")
    
    try:
        # Triangulate the point
        points_4d_hom = cv2.triangulatePoints(P1, P2, pt_l, pt_r) # Pass 2x1 arrays
    
        if points_4d_hom[3,0] == 0: 
            print("Warning: Triangulation resulted in w=0 for 4D point.")
            return None
            
        points_3d = points_4d_hom[:3,0] / points_4d_hom[3,0]
        
        # Basic sanity check on depth
        if points_3d[2] < 0 or points_3d[2] > 10000:  # Negative or extremely large depth is likely an error
            print(f"Warning: Unrealistic depth value: {points_3d[2]}mm")
            return None
            
        return points_3d
        
    except Exception as e:
        print(f"Error in triangulation: {e}")
        return None

# --- Disparity Map Generation (SGBM) ---
def generate_disparity_map_sgbm(rectified_left_gray, rectified_right_gray):
    """Generates a disparity map using the SGBM algorithm."""
    if rectified_left_gray is None or rectified_right_gray is None:
        print("Error: Grayscale images for disparity are None.")
        return None, None

    # SGBM Parameters
    # These parameters may need tuning for optimal results.
    min_disparity = 0
    num_disparities = 64  # Must be divisible by 16
    block_size = 5       # Must be odd
    P1 = 8 * 3 * block_size**2  # 8*number_of_image_channels*blockSize*blockSize
    P2 = 32 * 3 * block_size**2 # 32*number_of_image_channels*blockSize*blockSize
    disp12_max_diff = 1
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    pre_filter_cap = 63 # Default is 63
    mode = cv2.STEREO_SGBM_MODE_SGBM # Default mode

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=mode
    )

    try:
        disparity_map_raw = stereo_sgbm.compute(rectified_left_gray, rectified_right_gray).astype(np.float32) / 16.0
        # Normalize the disparity map for display
        disparity_map_normalized = cv2.normalize(disparity_map_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return disparity_map_raw, disparity_map_normalized # Return both raw and normalized
    except Exception as e:
        print(f"Error computing disparity map: {e}")
        return None, None

# --- 3D from Disparity (using Q matrix) ---
def get_3d_from_disparity(u, v, disparity_value, Q):
    """Calculates 3D coordinates from a disparity value using the Q matrix."""
    if disparity_value <= 0:
        # print("Warning: Non-positive disparity value, cannot calculate 3D point.")
        return None

    # Create a 4D point (u, v, disparity, 1)
    point_4d = np.array([[u], [v], [disparity_value], [1.0]], dtype=np.float32)
    
    # Reproject to 3D using the Q matrix
    # Q is typically: [[1, 0, 0, -cx],
    #                 [0, 1, 0, -cy],
    #                 [0, 0, 0,  f],
    #                 [0, 0, 1/Tx, (cx - cx')/Tx]]
    # where (cx, cy) is principal point of left cam, f is focal length, Tx is baseline.
    # Result is [X*W, Y*W, Z*W, W]^T where W = disparity * (1/Tx)
    
    try:
        coords_3d_homogeneous = Q @ point_4d
        if coords_3d_homogeneous[3,0] == 0:
            print("Warning: Homogeneous W is 0 after Q matrix multiplication.")
            return None
        
        coords_3d = coords_3d_homogeneous[:3,0] / coords_3d_homogeneous[3,0]
        
        # Sanity check Z coordinate (depth)
        if coords_3d[2] <= 0 or coords_3d[2] > 20000: # Max depth 20 meters
            # print(f"Warning: Unrealistic Z from disparity: {coords_3d[2]}")
            return None
            
        return coords_3d
    except Exception as e:
        print(f"Error in get_3d_from_disparity: {e}")
        return None

# --- Main Application ---
if __name__ == "__main__":
    # Create OpenCV windows first
    window_left_name = "Rectified Left"
    window_right_name = "Rectified Right"
    window_disparity_name = "Disparity Map (SGBM)"
    try:
        cv2.namedWindow(window_left_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_right_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_disparity_name, cv2.WINDOW_NORMAL)
        print(f"OpenCV windows '{window_left_name}', '{window_right_name}', and '{window_disparity_name}' created.")
    except Exception as e:
        print(f"Error creating OpenCV windows: {e}")
        exit()

    intrinsics_path = "cal_results/intrinsics.yml"
    extrinsics_path = "cal_results/extrinsics.yml"

    params = load_calibration_files(intrinsics_path, extrinsics_path)
    if params is None: exit()
    K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q = params
    print("Calibration parameters loaded.")

    yolo_model = None # Renamed from yolo_net

    # Convert TARGET_CLASSES list to a set for faster lookups during detection
    target_classes_set = set(TARGET_CLASSES)

    try:
        from picamera2 import Picamera2 # Added here
        cam0 = Picamera2(camera_num=0)
        cam1 = Picamera2(camera_num=1) # Re-enabled
        print("DEBUG: cam0 and cam1 objects created.")
        config0 = cam0.create_preview_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT), "format": "RGB888"})
        config1 = cam1.create_preview_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT), "format": "RGB888"}) # Re-enabled
        print("DEBUG: config0 and config1 created.")
        cam0.configure(config0)
        cam1.configure(config1) # Re-enabled
        print("DEBUG: cam0 and cam1 configured.")
        cam0.start()
        cam1.start() # Re-enabled
        print("DEBUG: cam0 and cam1 started.")
        time.sleep(1)
        print("Picamera2 cameras 0 and 1 initialized and started.")
    except Exception as e:
        print(f"Error initializing Picamera2 for stereo: {e}"); exit() # Updated error message

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.CV_32FC1)
    print("Rectification maps computed.")
    
    last_time = time.time()
    print("DEBUG: last_time initialized.")
    loop_count = 0
    # windows_created = False # Flag no longer needed
    print("DEBUG: Entering main loop...")
    while True:
        loop_count += 1
        # print(f"DEBUG: Loop iteration {loop_count}")

        if yolo_model is None:
            print("DEBUG: First loop iteration, loading YOLO model...")
            yolo_model = load_yolo_model(MODEL_DIR_PATH) # Call updated load_yolo_model
            if yolo_model is None:
                print("ERROR: Failed to load YOLO model in loop. Exiting.")
                break
            print("DEBUG: YOLO model loaded successfully in loop.")

        # print("DEBUG: Before cam0.capture_array()")
        frame_left_rgb = cam0.capture_array() 

        # print("DEBUG: Before cam1.capture_array()") # Re-enabled print
        frame_right_rgb = cam1.capture_array() # Re-enabled capture
        # print("DEBUG: After cam1.capture_array()") # Re-enabled print
        
        # Simulate right frame to avoid error if cam1 capture is commented
        # if 'frame_right_rgb' not in locals(): # This will now always be true
        # frame_right_rgb = frame_left_rgb.copy() # Use copy of left frame for right # This line should be removed or commented

        # print("DEBUG: Before cvtColor left")
        frame_left_bgr = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)
        # print("DEBUG: After cvtColor left")

        # print("DEBUG: Before cvtColor right")
        frame_right_bgr = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)
        # print("DEBUG: After cvtColor right")

        # print("DEBUG: Before remap left")
        rectified_left = cv2.remap(frame_left_bgr, map1_l, map2_l, cv2.INTER_LINEAR)
        # print("DEBUG: After remap left")

        # print("DEBUG: Before remap right")
        rectified_right = cv2.remap(frame_right_bgr, map1_r, map2_r, cv2.INTER_LINEAR)
        # print("DEBUG: After remap right")
        
        # Convert rectified images to grayscale for disparity calculation
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        
        # Generate and display disparity map
        disparity_map_raw, disparity_map_normalized = generate_disparity_map_sgbm(gray_left, gray_right)
        if disparity_map_raw is not None and disparity_map_normalized is not None:
            cv2.imshow(window_disparity_name, disparity_map_normalized)
        
        # Call to detect_objects needs to pass the correct model and target_classes_set
        detections_left = detect_objects(rectified_left.copy(), yolo_model, target_classes_set,
                                              MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, # input_width/height may become obsolete
                                              CONFIDENCE_THRESHOLD)
        
        detections_right = detect_objects(rectified_right.copy(), yolo_model, target_classes_set,
                                               MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, # input_width/height may become obsolete
                                               CONFIDENCE_THRESHOLD)
        
        for det_l in detections_left:
            x_l, y_l, w_l, h_l = det_l["box_xywh"]
            center_l = (x_l + w_l // 2, y_l + h_l // 2)
            cv2.rectangle(rectified_left, (x_l, y_l), (x_l + w_l, y_l + h_l), (255, 0, 0), 2)
            cv2.putText(rectified_left, f"{det_l['class_name']}: {det_l['confidence']:.2f}",
                        (x_l, y_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            best_match_right_center = None
            min_y_diff = 40  # Increased from 20 to 40 to accommodate larger vertical differences
            
            # Add debug prints
            print(f"Left detection: {det_l['class_name']} at {center_l}")
            
            for det_r in detections_right:
                if det_r["class_name"] == det_l["class_name"]:
                    x_r, y_r, w_r, h_r = det_r["box_xywh"]
                    center_r = (x_r + w_r // 2, y_r + h_r // 2)
                    y_difference = abs(center_l[1] - center_r[1])
                    
                    # Debug right detection
                    print(f"  Potential match: {det_r['class_name']} at {center_r}, y_diff={y_difference}")
                    
                    # In rectified stereo, corresponding points should have similar y-coordinates
                    # We need to ensure objects in both views match properly
                    if y_difference < min_y_diff:
                        min_y_diff = y_difference
                        best_match_right_center = center_r
                        print(f"  Found match: {center_r}")
                        # Draw matched detection on right image for visualization
                        cv2.rectangle(rectified_right, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 0, 255), 2) 
                        cv2.putText(rectified_right, f"{det_r['class_name']}", (x_r, y_r -5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)

            if best_match_right_center:
                print(f"Calling get_3d_coordinates with {center_l} and {best_match_right_center}")
                coords_3d_dlt = get_3d_coordinates(center_l, best_match_right_center, P1, P2)
                if coords_3d_dlt is not None:
                    print(f"DLT 3D coordinates: {coords_3d_dlt}")
                    label_3d_dlt = f"DLT: X{coords_3d_dlt[0]:.0f} Y{coords_3d_dlt[1]:.0f} Z{coords_3d_dlt[2]:.0f}mm"
                    cv2.putText(rectified_left, label_3d_dlt, (x_l, y_l - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                else:
                    print("get_3d_coordinates (DLT) returned None")
            else:
                print("No matching DLT detection found in right image for 3D calculation")

            # Calculate 3D coordinates using SGBM disparity map
            coords_3d_sgbm = None
            if disparity_map_raw is not None:
                # Ensure center_l coordinates are within the disparity map bounds
                if 0 <= center_l[1] < disparity_map_raw.shape[0] and 0 <= center_l[0] < disparity_map_raw.shape[1]:
                    disparity_at_center_l = disparity_map_raw[center_l[1], center_l[0]]
                    if disparity_at_center_l > 0:
                        # print(f"Disparity at {center_l} for {det_l['class_name']}: {disparity_at_center_l}")
                        coords_3d_sgbm = get_3d_from_disparity(center_l[0], center_l[1], disparity_at_center_l, Q)
                        if coords_3d_sgbm is not None:
                            print(f"SGBM 3D coordinates for {det_l['class_name']} at {center_l} (Disparity: {disparity_at_center_l:.2f}): {coords_3d_sgbm}")
                            label_3d_sgbm = f"SGBM: X{coords_3d_sgbm[0]:.0f} Y{coords_3d_sgbm[1]:.0f} Z{coords_3d_sgbm[2]:.0f}mm"
                            cv2.putText(rectified_left, label_3d_sgbm, (x_l, y_l - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1) # New Y offset and color
                        # else:
                            # print(f"SGBM 3D calculation failed for {det_l['class_name']} at {center_l}")
                    # else:
                        # print(f"Non-positive disparity ({disparity_at_center_l}) at {center_l} for {det_l['class_name']}. Cannot calculate SGBM 3D.")
                # else:
                    # print(f"Center_l {center_l} out of bounds for disparity_map_raw shape {disparity_map_raw.shape}")
            # else:
                # print("Disparity map (raw) is None, cannot calculate SGBM 3D.")

        for i in range(20, IMAGE_HEIGHT, 40):
            cv2.line(rectified_left, (0, i), (IMAGE_WIDTH, i), (0, 255, 0), 1)
            cv2.line(rectified_right, (0, i), (IMAGE_WIDTH, i), (0, 255, 0), 1)

        # print("DEBUG: Before FPS calculation")
        current_time = time.time()
        fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        last_time = current_time
        cv2.putText(rectified_left, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        # print("DEBUG: Before imshow Left")
        cv2.imshow(window_left_name, rectified_left) # imshow will create a default window
        # print("DEBUG: After imshow Left")

        # print("DEBUG: Before imshow Right")
        cv2.imshow(window_right_name, rectified_right) # imshow will create a default window
        # print("DEBUG: After imshow Right")

        # print("DEBUG: Before waitKey")
        key = cv2.waitKey(1) & 0xFF
        # print(f"DEBUG: After waitKey, key={key}")
        if key == ord('q'): break
    
    # if 'cam1' in locals() and cam1.started: # Check if cam1 was ever defined and started
    #    cam1.stop() # Temporarily disable cam1
    #    cam1.close()
    # Ensure cam1 is stopped and closed if it was initialized
    if 'cam1' in locals():
        try:
            if cam1.started: # Check if cam1 object exists and was started
                cam1.stop()
            cam1.close() # Close it regardless if it was started or just initialized
            print("DEBUG: cam1 stopped and closed.")
        except Exception as e:
            print(f"DEBUG: Error stopping/closing cam1: {e}")

    cam0.close()
    cv2.destroyAllWindows()
    print("Script finished.") 