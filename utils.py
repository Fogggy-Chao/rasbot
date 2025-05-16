from gpiozero import Motor, Device, PWMOutputDevice, Servo
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero.pins.pigpio import PiGPIOFactory
import time
# Use PiGPIO for more precise PWM (optional but recommended)
factory = LGPIOFactory()
Device.pin_factory = factory

# Define GPIO pins
PWM_PIN = 18    # PWM pin for motor speed control
AIN1_PIN = 23   # Direction control 1
AIN2_PIN = 24   # Direction control 2

# Create global motor object
motor_1 = None
motor_2 = None
motor_3 = None
motor_4 = None
pwm = None
servo = None

def initialize_motor():
    """Initialize motor and PWM, return them in a dictionary."""
    global motor_1
    global pwm
    
    pwm = PWMOutputDevice(PWM_PIN, frequency=10000)
    motor_1 = Motor(AIN1_PIN, AIN2_PIN)
    # motor_2 = Motor(BIN1_PIN, BIN2_PIN)
    # motor_3 = Motor(CIN1_PIN, CIN2_PIN)
    # motor_4 = Motor(DIN1_PIN, DIN2_PIN)
    
    # Stop the motor (sets both pins to LOW)
    motor_1.stop()
    pwm.off()
    
    print("Motor and PWM initialized")
    return {"motor_1": motor_1, "pwm": pwm} # Modified to return both

def motor_update(data):
    """
    Update the motor state based on the received data
    """
    global motor_1
    # global motor_2 # Not used yet
    # global motor_3 # Not used yet
    # global motor_4 # Not used yet
    global pwm # Added global pwm here for clarity, as it's used
    print(f"Received motor update: {data}")
    
    # If motor hasn't been initialized, do it now
    # This check might be redundant if main.py always initializes first
    if motor_1 is None or pwm is None:
        print("Warning: motor_update called before initialization.")
        # Optionally, initialize here, or just return an error/warning
        init_objects = initialize_motor()
        # Note: initialize_motor() sets globals, so no need to reassign from init_objects here for globals
        if motor_1 is None or pwm is None: # Check again if init failed
             print("Error: Initialization failed in motor_update. Cannot proceed.")
             return
    
    motor1_active = data.get("motor1", False) # Renamed to avoid conflict with global motor1
    # motor2 = data.get("motor2", False)
    # motor3 = data.get("motor3", False)
    # motor4 = data.get("motor4", False)
    m1_speed = data.get("m1_speed", .1)
    # m2_speed = data.get("m2_speed", .1)
    # m3_speed = data.get("m3_speed", .1)
    # m4_speed = data.get("m4_speed", .1)

    # Validate and adjust motor speeds to be within 0-1 range
    if not (0 <= m1_speed <= 1): # Simplified for only m1_speed for now
        print(f"Invalid speed for motor1: {m1_speed}, using default value 0.1")
        m1_speed = 0.1
    
    print(f'Motor 1 Active: {motor1_active} Speed: {m1_speed}')

    # Control Motor 1
    if motor1_active:
        print("Motor 1 is on")
        motor_1.forward(m1_speed) # Use global motor_1
        pwm.on()                  # Use global pwm
    else:
        print("Motor 1 is off")
        # Stop the motor
        motor_1.stop() # Use global motor_1
        pwm.off()      # Use global pwm

# Function to clean up on program exit
def motor_cleanup():
    """Clean up resources"""
    global motor_1
    # global motor_2
    # global motor_3
    # global motor_4
    global pwm # Added global pwm here for clarity

    if motor_1 is not None:
        motor_1.stop() # Use global motor_1
        # gpiozero automatically cleans up GPIO resources
    if pwm is not None:
        pwm.off()      # Use global pwm

    print("Motor cleaned up")

def initialize_servo():
    """Initialize servo"""
    global servo
    servo = Servo(PWM_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
    servo.detach()
    return servo

def pwm_servo():
    """
    Test servo movement by cycling through minimum, maximum, and center positions
    """
    global servo
    # pwm = PWMOutputDevice(PWM_PIN, frequency=100)
    try:
        # Turn on PWM once before the loop
        print("Servo on", servo.is_active)
        while True:
            # Minimum position (approximately 0 degrees)
            # Standard servo minimum is ~1ms pulse, which is ~5% duty at 50Hz
            servo.min()
            print("Servo at minimum position")
            time.sleep(1)
            
            # Center position (approximately 90 degrees)
            # Standard servo center is ~1.5ms pulse, which is ~7.5% duty at 50Hz
            # servo.mid()
            # print("Servo at center position")
            # time.sleep(1)
            
            # Maximum position (approximately 180 degrees)
            # Standard servo maximum is ~2ms pulse, which is ~10% duty at 50Hz
            servo.max()
            print("Servo at maximum position")
            time.sleep(1)
            
    except KeyboardInterrupt:
        servo.close()
        print("Servo cleanup")
    finally:
        # Ensure PWM is turned off even if exception occurs
        print("Servo detached")
