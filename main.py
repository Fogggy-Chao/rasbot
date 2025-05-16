import asyncio
from wss import start_server
from utils import initialize_motor, motor_cleanup
from servo import init_robot_arm, reset_to_home
# Import vision system components
from vision_system import initialize_vision, cleanup_vision
import os

# Define paths for vision system initialization
# These paths are relative to the location of main.py (i.e., inside the 'rabot' directory)
INTRINSICS_FILE_PATH = "cam_calibration/cal_results/intrinsics.yml"
EXTRINSICS_FILE_PATH = "cam_calibration/cal_results/extrinsics.yml"
YOLO_MODEL_DIR_PATH = "yolo/yolo11n_ncnn_model/"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

async def main():
    robot_contexts = {
        "vision_system": None, # For the whole vision context dictionary
        "servo_kit": None,
        "motor_1": None,
        "pwm_motor": None,
        "keep_running": True
    }

    try:
        print("Initializing robot components...")
        
        # 1. Initialize Motors
        motor_objects = initialize_motor()
        robot_contexts["motor_1"] = motor_objects.get("motor_1")
        robot_contexts["pwm_motor"] = motor_objects.get("pwm")
        print("Motors initialized.")

        # 2. Initialize Servo Arm
        # robot_contexts["servo_kit"] = init_robot_arm()
        # print("Servo arm initialized.")

        # 3. Initialize Vision System
        print("Initializing vision system...")
        # Ensure paths are absolute or correctly relative for vision_system module
        # os.path.abspath can be useful if vision_system.py expects absolute paths
        # However, initialize_vision in vision_system.py already uses os.path.abspath for yolo_model_dir_path.
        # For calibration files, it uses them as passed.
        # Let's ensure they are robust by making them absolute from main.py's perspective.
        
        # script_dir = os.path.dirname(__file__) # Get directory of main.py
        # abs_intrinsics_path = os.path.join(script_dir, INTRINSICS_FILE_PATH)
        # abs_extrinsics_path = os.path.join(script_dir, EXTRINSICS_FILE_PATH)
        # abs_yolo_model_path = os.path.join(script_dir, YOLO_MODEL_DIR_PATH)
        # Using direct relative paths assuming main.py is in rabot/ and those subdirs exist

        vision_ctx = initialize_vision(
            intrinsics_path=INTRINSICS_FILE_PATH,
            extrinsics_path=EXTRINSICS_FILE_PATH,
            yolo_model_dir_path=YOLO_MODEL_DIR_PATH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT
        )
        if vision_ctx:
            robot_contexts["vision_system"] = vision_ctx
            print("Vision system initialized successfully.")
        else:
            print("ERROR: Vision system initialization failed. Continuing without full vision capabilities.")
            # Decide if the application should exit or run with degraded functionality
            # For now, it will continue, and handle_detect_object will fail gracefully.
        
        print("All robot components initialized.")

        await start_server(robot_contexts)

    except KeyboardInterrupt:
        print("\nShutting down server and robot components by user...")
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        print("Cleaning up resources...")
        robot_contexts["keep_running"] = False

        if robot_contexts["vision_system"]:
            cleanup_vision(robot_contexts["vision_system"])
            print("Vision system cleanup called.")

        if robot_contexts["servo_kit"]:
            try:
                reset_to_home(robot_contexts["servo_kit"])
                print("Servos reset to home position.")
            except Exception as e:
                print(f"Error resetting servos: {e}")
        
        motor_cleanup()
        print("Motor cleanup called.")
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())