# vision_system.py
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
# We'll need Picamera2, but its import should be conditional or handled gracefully
# if not available on a non-Pi dev environment. For now, direct import.
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None # Placeholder if not on Pi
    print("WARN: Picamera2 not found. Live camera functionality will be disabled.")

# --- Constants (can be adjusted or passed as arguments if needed) ---
# These were in stereo_undistort_rectify.py, might need to be sourced from context or config
# IMAGE_WIDTH = 640
# IMAGE_HEIGHT = 480
# MODEL_INPUT_WIDTH = 640
# MODEL_INPUT_HEIGHT = 640
# CONFIDENCE_THRESHOLD = 0.4
# TARGET_CLASSES = ["cup", "bottle", "bowl", "apple"] # Example, will come from AI

# Define SAVE_FRAMES_PATH relative to this script's location
# This assumes vision_system.py is in the 'rabot' directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FRAMES_PATH = os.path.join(SCRIPT_DIR, "saved_detection_frames") # Changed path definition
MAX_SAVED_FRAME_PAIRS_PER_CALL = 3

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
    calibration_params = {}
    calibration_params['K1'] = fs_intrinsics.getNode("M1").mat()
    calibration_params['D1'] = fs_intrinsics.getNode("D1").mat()
    calibration_params['K2'] = fs_intrinsics.getNode("M2").mat()
    calibration_params['D2'] = fs_intrinsics.getNode("D2").mat()

    # Extrinsics
    calibration_params['R'] = fs_extrinsics.getNode("R").mat()
    calibration_params['T'] = fs_extrinsics.getNode("T").mat()
    calibration_params['R1'] = fs_extrinsics.getNode("R1").mat()
    calibration_params['R2'] = fs_extrinsics.getNode("R2").mat()
    calibration_params['P1'] = fs_extrinsics.getNode("P1").mat()
    calibration_params['P2'] = fs_extrinsics.getNode("P2").mat()
    calibration_params['Q'] = fs_extrinsics.getNode("Q").mat()

    fs_intrinsics.release()
    fs_extrinsics.release()
    
    # Check if all crucial parameters were loaded
    essential_keys = ['K1', 'D1', 'K2', 'D2', 'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q']
    for key in essential_keys:
        if calibration_params.get(key) is None:
            print(f"Error: Failed to load essential calibration parameter '{key}'")
            return None
            
    return calibration_params

# --- Load YOLO Model (using Ultralytics) ---
def load_yolo_model_internal(model_path): # Renamed to avoid conflict if old file is still around
    """Loads the YOLO model and returns the model and its class names."""
    print(f"Loading YOLO model from: {model_path}")
    abs_model_path = os.path.abspath(model_path)
    print(f"  Absolute YOLO model path: {abs_model_path}")
    try:
        model = YOLO(abs_model_path, task='detect')
        loaded_class_names = model.names # Load class names from the model
        print("Ultralytics YOLO model loaded successfully.")
        print(f"  Model classes: {loaded_class_names}")
        return model, loaded_class_names
    except Exception as e:
        print(f"Error loading Ultralytics YOLO model: {e}")
        return None, None

# --- Main Initialization Function ---
def initialize_vision(
    intrinsics_path, 
    extrinsics_path, 
    yolo_model_dir_path, 
    image_width=640, 
    image_height=480
):
    """
    Initializes all vision system components: calibration, YOLO model, cameras, rectification maps.
    Returns a context dictionary or None if initialization fails.
    """
    print("Initializing Vision System...")
    vision_context = {"image_width": image_width, "image_height": image_height}

    # 1. Load Calibration Parameters
    print(f"Loading calibration files: Intrinsics='{intrinsics_path}', Extrinsics='{extrinsics_path}'")
    calib_params = load_calibration_files(intrinsics_path, extrinsics_path)
    if calib_params is None:
        print("ERROR: Failed to load calibration parameters.")
        return None
    vision_context["calibration_params"] = calib_params
    print("Calibration parameters loaded.")

    # 2. Load YOLO Model
    yolo_model, loaded_class_names = load_yolo_model_internal(yolo_model_dir_path)
    if yolo_model is None:
        print("ERROR: Failed to load YOLO model.")
        return None
    vision_context["yolo_model"] = yolo_model
    vision_context["loaded_class_names"] = loaded_class_names
    print("YOLO model loaded.")

    # 3. Initialize Cameras
    if Picamera2 is None:
        print("ERROR: Picamera2 library not available, cannot initialize cameras.")
        return None
    try:
        cam_left = Picamera2(camera_num=0)
        cam_right = Picamera2(camera_num=1)
        
        config_left = cam_left.create_preview_configuration(main={"size": (image_width, image_height), "format": "RGB888"})
        config_right = cam_right.create_preview_configuration(main={"size": (image_width, image_height), "format": "RGB888"})
        
        cam_left.configure(config_left)
        cam_right.configure(config_right)
        
        cam_left.start()
        cam_right.start()
        time.sleep(1) # Allow cameras to warm up
        vision_context["cameras"] = {"left": cam_left, "right": cam_right}
        print("Stereo cameras initialized and started.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Picamera2 stereo cameras: {e}")
        if 'cam_left' in locals() and cam_left.started: cam_left.stop(); cam_left.close()
        if 'cam_right' in locals() and cam_right.started: cam_right.stop(); cam_right.close()
        return None

    # 4. Compute Rectification Maps
    try:
        K1, D1 = calib_params['K1'], calib_params['D1']
        K2, D2 = calib_params['K2'], calib_params['D2']
        R1, P1 = calib_params['R1'], calib_params['P1']
        R2, P2 = calib_params['R2'], calib_params['P2']
        
        map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (image_width, image_height), cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (image_width, image_height), cv2.CV_32FC1)
        vision_context["rectification_maps"] = {
            "left_map1": map1_l, "left_map2": map2_l,
            "right_map1": map1_r, "right_map2": map2_r
        }
        print("Rectification maps computed.")
    except Exception as e:
        print(f"ERROR: Failed to compute rectification maps: {e}")
        cleanup_vision(vision_context) # Cleanup already initialized parts
        return None
        
    print("Vision System Initialized Successfully.")
    return vision_context

def capture_and_rectify(vision_context):
    """Captures frames from stereo cameras and rectifies them."""
    cameras = vision_context.get("cameras")
    rect_maps = vision_context.get("rectification_maps")
    if not cameras or not rect_maps:
        print("Error: Cameras or rectification maps not found in vision context.")
        return None, None

    try:
        frame_left_rgb = cameras["left"].capture_array()
        frame_right_rgb = cameras["right"].capture_array()

        frame_left_bgr = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)
        frame_right_bgr = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)

        rectified_left = cv2.remap(frame_left_bgr, rect_maps["left_map1"], rect_maps["left_map2"], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right_bgr, rect_maps["right_map1"], rect_maps["right_map2"], cv2.INTER_LINEAR)
        
        return rectified_left, rectified_right
    except Exception as e:
        print(f"Error during frame capture and rectification: {e}")
        return None, None

def detect_objects_in_image(
    image_bgr, 
    yolo_model, 
    loaded_class_names, 
    target_object_name, # Single target object name string
    model_input_width=640, # This will become unused in the yolo_model call below
    model_input_height=640, # This will become unused in the yolo_model call below
    confidence_threshold=0.4 
):
    """
    Performs object detection on a single image for a specific target object.
    Returns a list of detections for the target object.
    """
    detected_objects_for_target = []
    if yolo_model is None:
        return detected_objects_for_target

    try:
        # Ultralytics handles image resizing and normalization internally.
        # Removing explicit imgsz to let YOLO use the model's default.
        predictions = yolo_model(image_bgr, verbose=False) 
    except Exception as e:
        print(f"Error during Ultralytics model inference: {e}")
        return detected_objects_for_target

    if not predictions or not predictions[0].boxes:
        return detected_objects_for_target

    for box in predictions[0].boxes:
        try:
            confidence = float(box.conf.item())
            if confidence < confidence_threshold:
                continue

            class_id = int(box.cls.item())
            class_name = loaded_class_names.get(class_id, "Unknown")

            if class_name.lower() != target_object_name.lower():
                continue

            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            
            width = xmax - xmin
            height = ymax - ymin
            box_xywh = [xmin, ymin, width, height]
            center_x = xmin + width // 2
            center_y = ymin + height // 2

            detected_objects_for_target.append({
                "class_name": class_name,
                "confidence": confidence,
                "box_xywh": box_xywh,
                "center_xy": (center_x, center_y)
            })
        except Exception as e:
            print(f"Error processing a detection box: {e}")
            continue
            
    return detected_objects_for_target

def match_stereo_detections(detections_left, detections_right, y_diff_threshold=40):
    """
    Matches detections between left and right frames.
    Assumes detections_left and detections_right are for the *same* target object.
    Returns a list of matched pairs: {"center_left": (x,y), "center_right": (x,y), "confidence": avg_conf}
    """
    matched_pairs = []
    if not detections_left or not detections_right:
        return matched_pairs

    for det_l in detections_left:
        best_match_r_center = None
        min_y_diff = y_diff_threshold # Use the threshold as the initial minimum
        best_det_r_confidence = 0

        for det_r in detections_right:
            # Class names should already match as per detect_objects_in_image filtering
            y_difference = abs(det_l["center_xy"][1] - det_r["center_xy"][1])
            
            if y_difference < min_y_diff:
                # Prioritize stronger horizontal alignment for rectified images.
                # Could also consider x-disparity (center_l[0] > center_r[0] typically)
                # and object size similarity as additional matching criteria.
                min_y_diff = y_difference
                best_match_r_center = det_r["center_xy"]
                best_det_r_confidence = det_r["confidence"]
        
        if best_match_r_center:
            avg_confidence = (det_l["confidence"] + best_det_r_confidence) / 2.0
            matched_pairs.append({
                "center_left": det_l["center_xy"],
                "center_right": best_match_r_center,
                "confidence": avg_confidence 
            })
            # For simplicity, taking the first good match for the left detection.
            # More sophisticated matching could find the globally best pair.
            # Or ensure a right detection is only matched once.
            # For now, if multiple left detections match the same right one, it's possible.

    # Sort by confidence (highest first) if multiple matches are found
    matched_pairs.sort(key=lambda p: p["confidence"], reverse=True)
    return matched_pairs


def triangulate_points_dlt(point_left_xy, point_right_xy, P1, P2):
    """
    Triangulates 3D coordinates from 2D image points using DLT.
    P1, P2 are projection matrices from calibration_params.
    point_left_xy, point_right_xy are (x,y) tuples or lists.
    """
    # This function adapts get_3d_coordinates from stereo_undistort_rectify.py
    if point_left_xy is None or point_right_xy is None:
        # print("triangulate_points_dlt: point_left or point_right is None") # Minor: reduce print frequency
        return None

    pt_l = np.array([[point_left_xy[0]], [point_left_xy[1]]], dtype=np.float32)
    pt_r = np.array([[point_right_xy[0]], [point_right_xy[1]]], dtype=np.float32)
    
    try:
        points_4d_hom = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
    
        if points_4d_hom[3,0] == 0: 
            # print("Warning: Triangulation resulted in w=0 for 4D point.")
            return None
            
        points_3d = points_4d_hom[:3,0] / points_4d_hom[3,0]
        
        if points_3d[2] < 0 or points_3d[2] > 10000:  
            # print(f"Warning: Unrealistic depth value from DLT: {points_3d[2]}mm")
            return None
            
        return points_3d
        
    except Exception as e:
        # print(f"Error in DLT triangulation: {e}") # Minor: reduce print frequency
        return None

def detect_object_over_time(
    vision_context, 
    target_object_name, 
    duration_seconds=3,
    model_input_width=640, 
    model_input_height=640,
    confidence_threshold_detect=0.4,
    matching_y_diff_threshold=40
):
    """
    Detects a target object over a specified duration, returning the one with the highest confidence.
    Saves the first few pairs of captured stereo frames at the beginning of the call.
    """
    print(f"Starting object detection for '{target_object_name}' over {duration_seconds} seconds...")
    print(f"DEBUG: Attempting to save frames to absolute path: {os.path.abspath(SAVE_FRAMES_PATH)}") # Debug print
    
    try:
        os.makedirs(SAVE_FRAMES_PATH, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create directory for saving frames: {SAVE_FRAMES_PATH}. Error: {e}")

    best_detection = {"object_position": None, "confidence": 0.0}
    
    yolo_model = vision_context.get("yolo_model")
    loaded_class_names = vision_context.get("loaded_class_names")
    calib_params_dict = vision_context.get("calibration_params")

    if not yolo_model: return {**best_detection, "error": "Missing YOLO model"}
    if not loaded_class_names: return {**best_detection, "error": "Missing loaded class names"}
    if not calib_params_dict: return {**best_detection, "error": "Missing calibration_params"}

    P1 = calib_params_dict.get("P1")
    P2 = calib_params_dict.get("P2")
    if P1 is None: return {**best_detection, "error": "Missing P1 matrix"}
    if P2 is None: return {**best_detection, "error": "Missing P2 matrix"}

    start_time = time.time()
    frames_processed = 0
    saved_pairs_count = 0

    while (time.time() - start_time) < duration_seconds:
        frames_processed += 1
        rect_l, rect_r = capture_and_rectify(vision_context)
        
        if rect_l is None or rect_r is None:
            print(f"DEBUG: Frame {frames_processed}, capture_and_rectify failed. rect_l is None: {rect_l is None}, rect_r is None: {rect_r is None}") # Debug print
            time.sleep(0.1) 
            continue
        else:
            print(f"DEBUG: Frame {frames_processed}, capture_and_rectify succeeded.") # Debug print

        if saved_pairs_count < MAX_SAVED_FRAME_PAIRS_PER_CALL:
            try:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                filename_l = os.path.join(SAVE_FRAMES_PATH, f"initial_f{frames_processed:03d}_{timestamp_str}_left.png")
                filename_r = os.path.join(SAVE_FRAMES_PATH, f"initial_f{frames_processed:03d}_{timestamp_str}_right.png")
                print(f"DEBUG: Attempting to write left frame: {filename_l}") # Debug print
                write_success_l = cv2.imwrite(filename_l, rect_l)
                print(f"DEBUG: Attempting to write right frame: {filename_r}") # Debug print
                write_success_r = cv2.imwrite(filename_r, rect_r)
                
                if write_success_l and write_success_r:
                    print(f"    Saved initial frame pair: {filename_l}, {filename_r}")
                    saved_pairs_count += 1
                else:
                    print(f"Warning: cv2.imwrite failed for one or both frames. Left success: {write_success_l}, Right success: {write_success_r}")
            except Exception as e_save:
                print(f"Warning: Failed to save initial frame pair. Error: {e_save}")
        
        # Proceed with detection logic for the current frames
        detections_l = detect_objects_in_image(
            rect_l, yolo_model, loaded_class_names, target_object_name,
            model_input_width, model_input_height, confidence_threshold_detect
        )
        detections_r = detect_objects_in_image(
            rect_r, yolo_model, loaded_class_names, target_object_name,
            model_input_width, model_input_height, confidence_threshold_detect
        )

        if not detections_l or not detections_r: continue

        matched_pairs = match_stereo_detections(detections_l, detections_r, matching_y_diff_threshold)
        if not matched_pairs: continue
        
        current_best_match_in_frame = matched_pairs[0]
        coords_3d = triangulate_points_dlt(
            current_best_match_in_frame["center_left"], 
            current_best_match_in_frame["center_right"], 
            P1, P2
        )

        if coords_3d is not None:
            current_confidence = current_best_match_in_frame["confidence"]
            if current_confidence > best_detection["confidence"]:
                best_detection["object_position"] = [float(c) for c in coords_3d]
                best_detection["confidence"] = float(current_confidence)
        
    if best_detection["object_position"] is not None:
        print(f"Finished detection for '{target_object_name}'. Best found: Pos={best_detection['object_position']}, Conf={best_detection['confidence']:.2f} ({frames_processed} frames)")
    else:
        print(f"Finished detection for '{target_object_name}'. Object not found reliably. ({frames_processed} frames)")
        
    return best_detection

def cleanup_vision(vision_context):
    """Cleans up vision system resources, primarily cameras."""
    print("Cleaning up Vision System...")
    cameras = vision_context.get("cameras")
    if cameras:
        cam_left = cameras.get("left")
        cam_right = cameras.get("right")
        try:
            if cam_left and hasattr(cam_left, 'started') and cam_left.started:
                cam_left.stop()
            if cam_left and hasattr(cam_left, 'close'):
                cam_left.close()
            print("Left camera stopped and closed.")
        except Exception as e:
            print(f"Error cleaning up left camera: {e}")
        
        try:
            if cam_right and hasattr(cam_right, 'started') and cam_right.started:
                cam_right.stop()
            if cam_right and hasattr(cam_right, 'close'):
                cam_right.close()
            print("Right camera stopped and closed.")
        except Exception as e:
            print(f"Error cleaning up right camera: {e}")
    print("Vision System cleanup finished.")

# Example of how this module might be tested (if run directly)
if __name__ == '__main__':
    print("Testing Vision System Module...")
    # These paths would need to be correctly set for testing
    # Assumes calibration files are in ../cal_results relative to this script if it's in rabot/
    # And yolo model in ../yolo/...
    # For robust testing, use absolute paths or paths relative to a known root.
    
    # Path adjustments assuming this script is in 'rabot' directory
    # and 'cal_results', 'yolo' are siblings of 'rabot' OR 'main.py' is in 'rabot' root.
    # If running from `python -m rabot.vision_system` from workspace root:
    intrinsics_file_path = "rabot/cam_calibration/cal_results/intrinsics.yml"
    extrinsics_file_path = "rabot/cam_calibration/cal_results/extrinsics.yml"
    yolo_path = "rabot/yolo/yolo11n_ncnn_model/" # Directory for YOLOv8

    if not (os.path.exists(intrinsics_file_path) and os.path.exists(extrinsics_file_path) and os.path.exists(yolo_path)):
        print("ERROR: One or more paths for testing are invalid. Please check.")
        print(f"  Intrinsics: {os.path.abspath(intrinsics_file_path)}")
        print(f"  Extrinsics: {os.path.abspath(extrinsics_file_path)}")
        print(f"  YOLO Path: {os.path.abspath(yolo_path)}")
        exit()

    vision_context_test = initialize_vision(
        intrinsics_path=intrinsics_file_path,
        extrinsics_path=extrinsics_file_path,
        yolo_model_dir_path=yolo_path,
        image_width=640,
        image_height=480
    )

    if vision_context_test:
        print("Vision system initialized for test.")
        target_to_find = "apple"
        print(f"Starting timed detection test for: {target_to_find}")
        best_detection_result = detect_object_over_time(
            vision_context_test, 
            target_to_find, 
            duration_seconds=5, 
            confidence_threshold_detect=0.3
        )
        if best_detection_result["object_position"]:
            print(f"TEST RESULT - Best '{target_to_find}': Pos={best_detection_result['object_position']}, Conf={best_detection_result['confidence']:.2f}")
        else:
            print(f"TEST RESULT - '{target_to_find}' not found reliably during timed test.")
        
        cleanup_vision(vision_context_test)
    else:
        print("Failed to initialize vision system for test.") 