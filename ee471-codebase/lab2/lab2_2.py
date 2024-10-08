# les go

import sys
import os

# Add the 'classes' directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))

from Robot import Robot

def main():
    robot = Robot()
    joint_angles1 = [0, 0, 0, 0]
    joint_angles2 = [15, -45, -60, 90]
    joint_angles3 = [-90, 15, 30, -45]
    
    print(f"joint_configuration: {joint_angles1}")
    print(robot.get_fk(joint_angles1))
    print("\n")
    print(f"joint_configuration: {joint_angles2}")
    print(robot.get_fk(joint_angles2))
    print("\n")
    print(f"joint_configuration: {joint_angles3}")
    print(robot.get_fk(joint_angles3))

if __name__ == "__main__":
    main()