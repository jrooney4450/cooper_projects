#!/usr/bin/env python

from niryo_one_python_api.niryo_one_api import *
import rospy
import time
import numpy as np
#!/usr/bin/env python

from niryo_one_python_api.niryo_one_api import *
import rospy
import time
import numpy as np
import csv
from piano_niryo import *

rospy.init_node('niryo_one_sim_test')
n = NiryoOne()

print "--- Start"

# Read notes from csv file using function in piano_niryo.py
coords = read_coords('moonlight_sonata.csv')
coords = coords/100

# keyboard properties
z_keyboard_off = 0.181   # how much z-offset the pencil and keyboard position add
z_white_press = -0.019   # .01 measured
z_black_press = -0.013   # .005 measured

# parameters of end effector with respect to robot base that go within n.move_pose() function
yaw_correct = float(raw_input('Run MoCap, what is the value of yaw_correct: ' ))
x_yaw_correct = -np.sin(yaw_correct)
y_yaw_correct = -np.cos(yaw_correct)

x_off = float(raw_input('What is the value of x_off: '))
y_off = float(raw_input('what is the value of y_off: '))
z_off = 0.1512 + z_keyboard_off
r_off = 0
p_off = np.pi/2
yaw_off = 0

try:
    # move robot to space where it can accept the pencil
    n.move_pose(0, 0.23, z_off, r_off, p_off, yaw_off)
    n.change_tool(TOOL_GRIPPER_1_ID)
    n.open_gripper(TOOL_GRIPPER_1_ID, 300)
    raw_input('Place the Sharpie in the robot gripper? If so press enter...')
    n.close_gripper(TOOL_GRIPPER_1_ID, 300)

    # move to C-key with transform from robot base to keyboard C-key as obtained from MoCap system
    n.move_pose(x_off, y_off, z_off, r_off, p_off, yaw_off)

    # error correct from pencil tip to keyboard C-key
    x_del = float(raw_input('Run MoCap again! What is the value of x_del: '))
    x_off += x_del
    y_del = float(raw_input('What is the value of y_del: '))
    y_off += y_del
    z_del = 0
    z_off += z_del
    n.move_pose(x_off, y_off, z_off, r_off, p_off, yaw_off)
    time.sleep(1)

    # add addiitonal systematic error correction if present
    temp = 0.
    x_off += temp
    y_off += temp

    # run through notes of song as defined by coordinates in C-key
    for i in range(len(coords)):
        mag = np.sqrt(coords[i,1]**2 + coords[i,0]**2)
        phi = np.arctan2(coords[i,0],coords[i,1]) - yaw_correct
        n.move_pose(x_off + mag*np.sin(phi), y_off - mag*np.cos(phi), z_off + coords[i,2], r_off, p_off, yaw_off)
        if coords[i,2] <= 0:  # for white key press, if z-coord is 0
            n.move_pose(x_off + mag*np.sin(phi), y_off - mag*np.cos(phi), z_off + coords[i,2] + z_white_press, r_off, p_off, yaw_off)
        elif coords[i,2] > 0:  # for black key press, if z-coord is pos
            n.move_pose(x_off + mag*np.sin(phi), y_off - mag*np.cos(phi), z_off + coords[i,2] + z_black_press, r_off, p_off, yaw_off)
        time.sleep(1)
        n.move_pose(x_off + mag*np.sin(phi), y_off - mag*np.cos(phi), z_off + coords[i,2], r_off, p_off, yaw_off)

except NiryoOneException as e:
    print e

print "--- End"
