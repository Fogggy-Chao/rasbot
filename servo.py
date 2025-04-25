from adafruit_servokit import ServoKit
import time
END_EFFECTOR_PIN = 0
BASE_PIN = 15
SHOULDER_PIN = 14
ELBOW_PIN = 13
WRIST_PIN = 12
WRIST_ROTATION_PIN = 11

# End effector angle range: 40 - 100
# Base angle range: 0 - 180
# Shoulder angle range: 0 - 180
# Elbow angle range: 0 - 180
# Wrist angle range: 0 - 180
# Wrist rotation angle range: 0 - 180

def init_robot_arm():
    """
    Initialize the robot arm
    """
    kit = ServoKit(channels=16)

    # Set pulse width range for all servos
    for joint in [END_EFFECTOR_PIN, BASE_PIN, SHOULDER_PIN, ELBOW_PIN, WRIST_PIN, WRIST_ROTATION_PIN]:
        kit.servo[joint].set_pulse_width_range(500, 2500)
        print(f"Servo {joint} pulse width set to range {500} - {2500}")

    kit.servo[BASE_PIN].angle = 90
    kit.servo[SHOULDER_PIN].angle = 90
    kit.servo[ELBOW_PIN].angle = 90
    kit.servo[WRIST_PIN].angle = 90
    kit.servo[WRIST_ROTATION_PIN].angle = 90
    kit.servo[END_EFFECTOR_PIN].angle = 60

    return kit

def reset_to_home(kit):
    """Move all servos to their home/neutral positions"""
    # Define home positions
    kit.servo[BASE_PIN].angle = 90
    kit.servo[SHOULDER_PIN].angle = 90
    kit.servo[ELBOW_PIN].angle = 90
    kit.servo[WRIST_PIN].angle = 90
    kit.servo[WRIST_ROTATION_PIN].angle = 90
    kit.servo[END_EFFECTOR_PIN].angle = 30  # Middle of 0-60 range
    
    print("Robot arm reset to home position")

def wave_arm(kit, wave_count=2, wave_speed=1):
    """
    Make the arm wave by moving the waist/base servo side to side
    
    Parameters:
    - kit: ServoKit instance
    - wave_count: Number of complete waves (default: 2)
    - wave_speed: Speed of the wave in seconds (default: 1)
    """
    # Store the original position to return to
    elbow_original_position = kit.servo[ELBOW_PIN].angle
    base_original_position = kit.servo[BASE_PIN].angle
    
    # Define waist movement range (adjust these values for wider/narrower waving)
    left_position = 70   # Left position for waving
    right_position = 110  # Right position for waving
    
    # Wave motion
    kit.servo[BASE_PIN].angle = 0

    for _ in range(wave_count):
        # Move to left position
        # kit.servo[WRIST_PIN].angle = left_position
        move_servo_smooth(kit, WRIST_PIN, left_position)
        time.sleep(wave_speed)
        
        # Move to right position
        # kit.servo[WRIST_PIN].angle = right_position
        move_servo_smooth(kit, WRIST_PIN, right_position)
        time.sleep(wave_speed)
    
    # Return to original position
    kit.servo[WRIST_PIN].angle = elbow_original_position
    kit.servo[BASE_PIN].angle = base_original_position
    
def move_servo_smooth(kit, servo_pin, target_angle, steps=20, delay=0.02):
    """Move a servo smoothly from current position to target position"""
    current_angle = kit.servo[servo_pin].angle
    step_size = (target_angle - current_angle) / steps
    
    for i in range(steps):
        next_angle = current_angle + step_size * (i+1)
        kit.servo[servo_pin].angle = next_angle
        time.sleep(delay)
    
def handshake(kit, shake_count=3, shake_speed=0.5):
    """
    Perform a handshake gesture using wrist movement
    
    Parameters:
    - kit: ServoKit instance
    - shake_count: Number of shakes (default: 3)
    - shake_speed: Speed of each shake in seconds (default: 0.3)
    """
    # Store original positions
    wrist_original = kit.servo[WRIST_PIN].angle
    
    # Position arm for handshake
    move_servo_smooth(kit, BASE_PIN, 90)
    move_servo_smooth(kit, SHOULDER_PIN, 120)
    move_servo_smooth(kit, ELBOW_PIN, 90)
    move_servo_smooth(kit, WRIST_PIN, 90)
    #Close grip
    move_servo_smooth(kit, END_EFFECTOR_PIN, 60)
    time.sleep(0.5)
    
    # Shake hands by moving wrist up and down
    for _ in range(shake_count):
        # Move wrist up
        move_servo_smooth(kit, WRIST_PIN, 70)
        time.sleep(shake_speed)
        
        # Move wrist down
        move_servo_smooth(kit, WRIST_PIN, 110)
        time.sleep(shake_speed)
    
    # Return wrist to original position
    move_servo_smooth(kit, WRIST_PIN, wrist_original)
    
def look_around(kit):
    """Move the arm as if it's looking around"""
    # Store original positions
    base_original = kit.servo[BASE_PIN].angle
    
    # Look left, right, and center
    move_servo_smooth(kit, BASE_PIN, 30)  # Look left
    time.sleep(0.5)
    move_servo_smooth(kit, BASE_PIN, 150)  # Look right
    time.sleep(0.5)
    move_servo_smooth(kit, BASE_PIN, base_original)  # Look center
    
def bow(kit, bow_depth=70, pause_time=1.0):
    """
    Makes the robot arm perform a deeper, more respectful bow
    
    Parameters:
    - kit: ServoKit instance
    - bow_depth: How deep to bow (in degrees) (default: 70)
    - pause_time: How long to hold the bow (default: 1.0)
    """
    # Store original positions
    shoulder_original = kit.servo[SHOULDER_PIN].angle
    elbow_original = kit.servo[ELBOW_PIN].angle
    
    # Center position first
    move_servo_smooth(kit, BASE_PIN, 90)
    
    # Deeper bow motion - shoulder down more, elbow compensates
    move_servo_smooth(kit, SHOULDER_PIN, shoulder_original + bow_depth)
    move_servo_smooth(kit, ELBOW_PIN, max(0, elbow_original - bow_depth/1.5))
    
    # Hold the bow
    time.sleep(pause_time)
    
    # Return to original position with smooth movement
    move_servo_smooth(kit, SHOULDER_PIN, shoulder_original)
    move_servo_smooth(kit, ELBOW_PIN, elbow_original)
    

def dance(kit, moves=5, speed=0.3):
    """
    Makes the robot arm perform a dynamic dance routine without using the end effector
    
    Parameters:
    - kit: ServoKit instance
    - moves: Number of dance moves (default: 5)
    - speed: Speed of the dance (default: 0.3)
    """
    # Start in home position
    reset_to_home(kit)
    
    for i in range(moves):
        # Dynamic base movement
        base_angle = 45 + (i * 30) % 90
        move_servo_smooth(kit, BASE_PIN, base_angle)
        
        # Coordinated shoulder and elbow movement (like a wave)
        move_servo_smooth(kit, SHOULDER_PIN, 60)
        move_servo_smooth(kit, ELBOW_PIN, 120)
        time.sleep(speed)
        move_servo_smooth(kit, SHOULDER_PIN, 120)
        move_servo_smooth(kit, ELBOW_PIN, 60)
        time.sleep(speed)
        
        # Twist wrist in both directions
        move_servo_smooth(kit, WRIST_PIN, 60)
        move_servo_smooth(kit, WRIST_ROTATION_PIN, 30)
        time.sleep(speed)
        move_servo_smooth(kit, WRIST_PIN, 120)
        move_servo_smooth(kit, WRIST_ROTATION_PIN, 150)
        time.sleep(speed)
        
        # Change base angle for variety
        move_servo_smooth(kit, BASE_PIN, 135 - (i * 30) % 90)
    
    # Return to home position
    reset_to_home(kit)

def express_no(kit, intensity=3, speed=0.3):
    """
    Makes the robot arm express "no" through simple wrist rotation
    
    Parameters:
    - kit: ServoKit instance
    - intensity: How strongly to express disagreement (1-3) (default: 2)
    - speed: Speed of the gesture (default: 0.3)
    """
    # Store original position to return to
    original_position = kit.servo[WRIST_ROTATION_PIN].angle
    
    # Calculate motion range based on intensity
    rotation_range = 20 * intensity  # 20, 40, or 60 degrees to each side
    
    # Simple wrist rotation for "no" gesture
    for _ in range(intensity):
        # Move left
        move_servo_smooth(kit, WRIST_ROTATION_PIN, 
                         180,
                         steps=5, delay=speed/2)
        
        # Move right
        move_servo_smooth(kit, WRIST_ROTATION_PIN, 
                         0,
                         steps=5, delay=speed/2)
    
    # Return to original position
    move_servo_smooth(kit, WRIST_ROTATION_PIN, original_position)

def kinematics(kit, angles):
    """
    Test the position of the end effector based on the angles of the servos
    """
    move_servo_smooth(kit, BASE_PIN, angles[0], delay=0.1)
    move_servo_smooth(kit, SHOULDER_PIN, angles[1], delay=0.1)
    move_servo_smooth(kit, ELBOW_PIN, angles[2], delay=0.1)
    move_servo_smooth(kit, WRIST_PIN, angles[3], delay=0.1)
    move_servo_smooth(kit, WRIST_ROTATION_PIN, angles[4], delay=0.1)
    move_servo_smooth(kit, END_EFFECTOR_PIN, angles[5], delay=0.1)


kit = init_robot_arm()
kinematics(kit, 
    [0, 45, 45, 45, 45, 0]
    )
