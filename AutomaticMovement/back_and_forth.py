"""
back_and_forth.py — Pioneer 3-AT controller that drives forward and backward
every 2 seconds in a continuous loop.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")
sys.path.append(os.path.join(webots_home, "lib", "controller", "python"))

from controller import Robot  # type: ignore

MAX_SPEED = 6.28
DRIVE_SPEED = MAX_SPEED * 0.75
TOGGLE_INTERVAL = 0.75  # seconds before switching direction

os.environ["WEBOTS_CONTROLLER_URL"] = "Pioneer 3-AT"
def auto():
    os.environ["WEBOTS_CONTROLLER_URL"] = "Pioneer 3-AT"
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Pioneer 3-AT has 4 wheels
    motor_names = [
        "front left wheel",
        "front right wheel",
        "back left wheel",
        "back right wheel",
    ]
    motors = []
    for name in motor_names:
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)

    moving_forward = True
    elapsed = 0.0
    dt = timestep / 1000.0  # convert ms → seconds

    print("Pioneer 3-AT: driving back and forth every 5 seconds")

    while robot.step(timestep) != -1:
        elapsed += dt

        if elapsed >= TOGGLE_INTERVAL:
            moving_forward = not moving_forward
            elapsed = 0.0
            direction = "FORWARD" if moving_forward else "BACKWARD"
            print(f"Switching to {direction}")

        speed = DRIVE_SPEED if moving_forward else -DRIVE_SPEED
        for motor in motors:
            motor.setVelocity(speed)

if __name__ == "__main__":
    auto()