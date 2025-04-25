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
    """Initialize motor"""
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
    
    print("Motor initialized")
    return motor_1

def motor_update(data):
    """
    Update the motor state based on the received data
    """
    global motor_1
    global motor_2
    global motor_3
    global motor_4
    print(f"Received motor update: {data}")
    
    # If motor hasn't been initialized, do it now
    if motor_1 is None:
        motor_1 = initialize_motor()
    
    motor1 = data.get("motor1", False)
    motor2 = data.get("motor2", False)
    motor3 = data.get("motor3", False)
    motor4 = data.get("motor4", False)
    m1_speed = data.get("m1_speed", .1)
    m2_speed = data.get("m2_speed", .1)
    m3_speed = data.get("m3_speed", .1)
    m4_speed = data.get("m4_speed", .1)
    # Validate and adjust motor speeds to be within 0-1 range
    if not (0 <= m1_speed <= 1 and 0 <= m2_speed <= 1 and 0 <= m3_speed <= 1 and 0 <= m4_speed <= 1):
        print(f"Invalid speed for motors: {m1_speed}, {m2_speed}, {m3_speed}, {m4_speed}, using default value 0.1")
        m1_speed = 0.1
        m2_speed = 0.1
        m3_speed = 0.1
        m4_speed = 0.1
    
    print(f'''Motor 1: {motor1} Speed: {m1_speed}\n
             Motor 2: {motor2} Speed: {m2_speed}\n
             Motor 3: {motor3} Speed: {m3_speed}\n
             Motor 4: {motor4} Speed: {m4_speed}''')

    # Control Motor 1
    if motor1:
        print("Motor 1 is on")
        motor_1.forward(m1_speed)
        pwm.on()
    else:
        print("Motor 1 is off")
        # Stop the motor
        motor_1.stop()
        pwm.off()

# Function to clean up on program exit
def motor_cleanup():
    """Clean up resources"""
    global motor_1
    global motor_2
    global motor_3
    global motor_4

    if motor_1 is not None:
        motor_1.stop()
        # gpiozero automatically cleans up GPIO resources
    if pwm is not None:
        pwm.off()

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
