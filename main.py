import asyncio
from wss import start_server
from utils import initialize_motor, motor_cleanup, pwm_servo, initialize_servo

if __name__ == "__main__":
    try:
        motor_1 = initialize_motor()
        # servo = initialize_servo()
        # pwm_servo()
        asyncio.run(start_server())

    except KeyboardInterrupt:
        motor_cleanup()
        print("Server stopped by user")