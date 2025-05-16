# robot_interface.py
import time
# We'll need to import functions from servo, utils, and the refactored vision script later
import servo as robot_servo_control # Corrected import
# from . import utils as robot_motor_control # Example - not needed for current changes
# from .cam_calibration import stereo_undistort_rectify as robot_vision # Example
# Import functions from our new vision_system module
from vision_system import (
    # capture_and_rectify, # No longer directly called by handle_detect_object
    # detect_objects_in_image, 
    # match_stereo_detections, 
    # triangulate_points_dlt,
    detect_object_over_time # New main function to call
)
# Constants that might be needed by handle_detect_object
# These could also be part of vision_context if they vary, e.g. from model properties
# MODEL_INPUT_WIDTH = 640 # Default, should ideally match yolo_model's expectations
# MODEL_INPUT_HEIGHT = 640
# CONFIDENCE_THRESHOLD_DETECT = 0.4 # Default detection confidence
# MATCHING_Y_DIFF_THRESHOLD = 40 # Pixel difference for stereo matching
DETECTION_DURATION_SECONDS = 3 # Duration for the timed detection loop

# Placeholder for home position for inverse kinematics
HOME_JOINT_ANGLES = [90, 90, 90, 90, 90, 60] # Base, Shoulder, Elbow, Wrist, Wrist Rotation, End Effector (example)

# Placeholder for robot contexts (initialized hardware/models)
# This will be passed from main.py
# robot_contexts = {
# "yolo_model": None,
# "calibration_params": None,
# "cameras": None, # e.g. {"cam0": picam0, "cam1": picam1}
# "servo_kit": None,
# "motor_1": None,
# "pwm_motor": None
# }

# --- Tool Implementations ---

def handle_detect_object(object_name, contexts):
    """
    Run YOLO + stereo triangulation over a short duration to locate a target object reliably.
    Uses DLT for 3D coordinate calculation.
    """
    print(f"TOOL CALL: detect_object, object_name: '{object_name}' (will run for {DETECTION_DURATION_SECONDS}s)")
    vision_ctx = contexts.get("vision_system")
    if not vision_ctx:
        print("ERROR: Vision system context not available.")
        return {"object_position": None, "confidence": 0.0, "error": "Vision system not initialized"}

    # Call the new timed detection function from vision_system
    # It returns a dictionary like: {"object_position": [x,y,z] or None, "confidence": float}
    best_detection_result = detect_object_over_time(
        vision_context=vision_ctx,
        target_object_name=object_name,
        duration_seconds=DETECTION_DURATION_SECONDS,
        # model_input_width, model_input_height, confidence_threshold_detect, matching_y_diff_threshold
        # will use defaults in detect_object_over_time or can be passed if needed here.
        # For example, if these constants were part of robot_contexts or a config file:
        # confidence_threshold_detect=contexts.get("config", {}).get("vision_confidence_detect", 0.4)
    )
    
    # The result from detect_object_over_time is already in the desired format.
    if best_detection_result["object_position"] is not None:
        print(f"handle_detect_object: '{object_name}' found at {best_detection_result['object_position']} with conf {best_detection_result['confidence']:.2f}")
    else:
        print(f"handle_detect_object: '{object_name}' not found reliably after {DETECTION_DURATION_SECONDS}s.")
        # Ensure a consistent return structure even if not found by timed detection.
        # detect_object_over_time should already do this (return with None position and 0.0 conf)

    return best_detection_result # Directly return the result

def handle_inverse_kinematics(object_position, contexts):
    """
    Calculate joint angles. For now, returns a predefined home position
    as actual IK is handled by the host.
    """
    print(f"TOOL CALL: inverse_kinematics, object_position: {object_position}")
    # As per user instruction, this is mostly a passthrough or returns home
    return {"joint_angles": HOME_JOINT_ANGLES}

def handle_move_arm(joint_angles, contexts):
    """
    Move the arm to a given set of joint angles.
    The order of joint_angles is assumed to be: [BASE, SHOULDER, ELBOW, WRIST, WRIST_ROTATION, END_EFFECTOR]
    However, END_EFFECTOR is typically controlled by grasp(). This function will ignore the last angle if 6 are provided,
    or expect 5 angles for the main arm segments.
    """
    print(f"TOOL CALL: move_arm, joint_angles: {joint_angles}")
    kit = contexts.get("servo_kit")
    if not kit:
        return {"status": "error", "message": "Servo kit not available in contexts."}

    # Define servo pins in the order expected by joint_angles
    # BASE_PIN, SHOULDER_PIN, ELBOW_PIN, WRIST_PIN, WRIST_ROTATION_PIN
    servo_pins_map = [
        robot_servo_control.BASE_PIN,
        robot_servo_control.SHOULDER_PIN,
        robot_servo_control.ELBOW_PIN,
        robot_servo_control.WRIST_PIN,
        robot_servo_control.WRIST_ROTATION_PIN
    ]

    if not isinstance(joint_angles, list) or not (5 <= len(joint_angles) <= 6):
        return {"status": "error", "message": "joint_angles must be a list of 5 or 6 angles."}

    angles_to_set = joint_angles[:5] # Take the first 5 for the main arm segments

    # Validate angles (0-170 degrees for these 5 joints)
    for i, angle in enumerate(angles_to_set):
        if not (0 <= angle <= 170):
            return {"status": "error", "message": f"Angle for joint {i} ({angle}deg) is out of 0-170 range."}
    
    try:
        print(f"  ARM: Moving to angles: {angles_to_set}")
        # Using kinematics function from servo.py (if it takes 5 angles or can be adapted)
        # Or, set them individually using move_servo_smooth
        # For now, let's assume individual control for clarity with validation
        for i, angle in enumerate(angles_to_set):
            pin = servo_pins_map[i]
            robot_servo_control.move_servo_smooth(kit, pin, angle) 
            # Adding a small delay between servo moves can sometimes be beneficial
            time.sleep(0.1) # Small delay

        # If a 6th angle (end effector) is provided by AI, it should ideally use grasp().
        # We could print a warning or ignore it. For now, we explicitly only use the first 5.
        if len(joint_angles) == 6:
            print(f"  ARM_NOTE: 6th angle provided ({joint_angles[5]}) for end effector. Use 'grasp' tool for gripper.")

        return {"status": "ok"}
    except Exception as e:
        print(f"  ARM: Error moving arm: {e}")
        return {"status": "error", "message": f"Error moving arm: {str(e)}"}

def handle_grasp(joint_angle, contexts):
    """
    Control the end effector (gripper).
    """
    print(f"TOOL CALL: grasp, joint_angle: {joint_angle}")
    kit = contexts.get("servo_kit")
    if not kit:
        return {"status": "error", "message": "Servo kit not available in contexts."}

    # Validate joint_angle against safe limits (40-100 deg for end effector)
    if not (40 <= joint_angle <= 100):
        return {"status": "error", "message": f"End effector angle ({joint_angle}deg) out of 40-100 range."}

    try:
        print(f"  GRIPPER: Setting angle to {joint_angle}")
        robot_servo_control.move_servo_smooth(kit, robot_servo_control.END_EFFECTOR_PIN, joint_angle)
        return {"status": "ok"}
    except Exception as e:
        print(f"  GRIPPER: Error setting gripper: {e}")
        return {"status": "error", "message": f"Error setting gripper: {str(e)}"}

def handle_rotate_base(base_angle, contexts):
    """
    Rotate the robot arm's base.
    """
    print(f"TOOL CALL: rotate_base, base_angle: {base_angle}")
    kit = contexts.get("servo_kit")
    if not kit:
        return {"status": "error", "message": "Servo kit not available in contexts."}

    # Validate base_angle against safe limits (0-170 deg for base servo)
    if not (0 <= base_angle <= 170):
        return {"status": "error", "message": f"Base angle ({base_angle}deg) out of 0-170 range."}

    try:
        print(f"  ARM_BASE: Setting angle to {base_angle}")
        robot_servo_control.move_servo_smooth(kit, robot_servo_control.BASE_PIN, base_angle)
        return {"status": "ok"}
    except Exception as e:
        print(f"  ARM_BASE: Error rotating base: {e}")
        return {"status": "error", "message": f"Error rotating base: {str(e)}"}

def handle_move_base(coordinates, contexts):
    """
    Drive the four-wheel chassis.
    For now, speed is PWM duty cycle, and returns dummy 'ok'.
    """
    print(f"TOOL CALL: move_base, coordinates: {coordinates}")
    # TODO:
    # 1. Access motor_1, pwm_motor from contexts
    # 2. For now, as per user: "directly return a dummy data, i.e. ok first"
    # 3. Later: Implement logic to translate coordinates [x,y] in mm into motor commands.
    #    - This will involve deciding on a strategy (e.g., move forward by X, then turn, then move by Y; or more complex path).
    #    - Wheel speed is PWM (0-1). The AI provides coordinates, not speed directly for this tool.
    #      We might need to assume a default speed or derive it.
    # from . import utils as robot_motor_control # Assuming utils is imported as robot_motor_control
    # robot_motor_control.motor_update({"motor1": True, "m1_speed": 0.5}) # Example forward
    # time.sleep(calculated_time_for_x)
    # robot_motor_control.motor_update({"motor1": False}) # Stop
    # ... then turn, then move for y ...
    print(f"  CHASSIS: Received move_base to {coordinates}. Dummy 'ok' returned for now.")
    return {"status": "ok"}

def handle_stop(contexts):
    """
    Stop the robot (chassis motors).
    """
    print(f"TOOL CALL: stop")
    motor = contexts.get("motor_1")
    pwm = contexts.get("pwm_motor")

    if motor and pwm:
        try:
            motor.stop()
            pwm.off()
            print("  CHASSIS: Motors stopped successfully.")
            return {"status": "ok"}
        except Exception as e:
            print(f"  CHASSIS: Error stopping motors: {e}")
            return {"status": "error", "message": f"Error stopping motors: {str(e)}"}
    else:
        missing_components = []
        if not motor: missing_components.append("motor_1")
        if not pwm: missing_components.append("pwm_motor")
        message = f"Motor context not fully available for stop. Missing: {', '.join(missing_components)}"
        print(f"  CHASSIS: {message}")
        return {"status": "error", "message": message}

# --- Main Dispatcher ---
def execute_tool_call(tool_name, arguments, contexts):
    """
    Dispatcher function to call the appropriate tool handler.
    """
    print(f"Executing tool: {tool_name} with arguments: {arguments}")
    if tool_name == "detect_object":
        return handle_detect_object(arguments.get("object_name"), contexts)
    elif tool_name == "inverse_kinematics":
        return handle_inverse_kinematics(arguments.get("object_position"), contexts)
    elif tool_name == "move_arm":
        # Ensure joint_angles is a list, handle potential errors if not provided correctly
        joint_angles_arg = arguments.get("joint_angles")
        if not isinstance(joint_angles_arg, list):
            print(f"Error: 'joint_angles' not a list or not provided for move_arm. Got: {joint_angles_arg}")
            return {"status": "error", "message": "Invalid or missing 'joint_angles'"}
        return handle_move_arm(joint_angles_arg, contexts)
    elif tool_name == "grasp":
        return handle_grasp(arguments.get("joint_angle"), contexts)
    elif tool_name == "rotate_base":
        return handle_rotate_base(arguments.get("base_angle"), contexts)
    elif tool_name == "move_base":
        return handle_move_base(arguments.get("coordinates"), contexts)
    elif tool_name == "stop":
        return handle_stop(contexts)
    else:
        print(f"Error: Unknown tool name '{tool_name}'")
        return {"status": "error", "message": f"Unknown tool: {tool_name}"} 