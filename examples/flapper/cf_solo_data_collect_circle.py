"""
qfly | Qualisys Drone SDK Example Script: Solo Crazyflie

Takes off, flies circles around Z, Y, X axes.
ESC to land at any time.
"""

import json
from datetime import datetime
from functools import partial
from time import sleep, time

import numpy as np
import pynput
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

from qfly import Pose, QualisysCrazyflie, World, utils

# SETTINGS
cf_body_name = "flapper"  # QTM rigid body name
cf_uri = "radio://0/80/2M/E7E7E7E7E7"  # Crazyflie address
cf_marker_ids = [1, 2, 3, 4]  # Active marker IDs
circle_radius = 0.75  # Radius of the circular flight path
circle_axis = "XYZ"  # Axis to circle around: 'X' or 'Y' or 'Z' or 'XYZ'
circle_speed_factor = 9  # How fast the Crazyflie should move along circle [degree/s]
qtm_ip = "128.174.245.190"

sampling_rate = 0.1  # sec

# Watch key presses with a global variable
last_key_pressed = None


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key
    if key == pynput.keyboard.Key.esc:
        fly = False


def log_callback(timestamp, data, logconf, data_log, key):
    print(f"{timestamp}, {data}, {logconf.name}")
    data_log[key].append(data)


# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()

# Set up world - the World object comes with sane defaults
world = World(expanse=1.8, speed_limit=1.1)

# Set up the asynchronous log configuration
# For details, see https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/
conf_list = []
group_list = [
    "stabilizer",
    "pos",
    "vel",
    "acc",
    "attitude_rate",
    "motor",
    "motor_req",
    "gyro",
    "target_pos",
    "target_vel",
    "controller_cmd",
    "controller_attitude",
    "controller_attitude_rate",
]
for group in group_list:
    logconf = LogConfig(name=group, period_in_ms=sampling_rate * 1000)
    if group == "stabilizer":
        logconf.add_variable("stabilizer.roll", "float")  # Same as stateEstimate.roll
        logconf.add_variable("stabilizer.pitch", "float")  # Same as stateEstimate.pitch
        logconf.add_variable("stabilizer.yaw", "float")  # Same as stateEstimate.yaw
        logconf.add_variable("stabilizer.thrust", "float")  # Current thrust
    if group == "pos":
        logconf.add_variable("stateEstimate.x", "float")
        logconf.add_variable("stateEstimate.y", "float")
        logconf.add_variable("stateEstimate.z", "float")
    if group == "vel":
        logconf.add_variable("stateEstimate.vx", "float")
        logconf.add_variable("stateEstimate.vy", "float")
        logconf.add_variable("stateEstimate.vz", "float")
    if group == "acc":
        logconf.add_variable("stateEstimate.ax", "float")
        logconf.add_variable("stateEstimate.ay", "float")
        logconf.add_variable("stateEstimate.az", "float")
    if group == "attitude_rate":
        logconf.add_variable("stateEstimateZ.rateRoll", "float")
        logconf.add_variable("stateEstimateZ.ratePitch", "float")
        logconf.add_variable("stateEstimateZ.rateYaw", "float")
    if group == "motor":
        logconf.add_variable("motor.m1", "float")
        logconf.add_variable("motor.m2", "float")
        logconf.add_variable("motor.m3", "float")
        logconf.add_variable("motor.m4", "float")
    if group == "motor_req":
        logconf.add_variable("motor.m1req", "float")
        logconf.add_variable("motor.m2req", "float")
        logconf.add_variable("motor.m3req", "float")
        logconf.add_variable("motor.m4req", "float")
    if group == "gyro":
        logconf.add_variable("gyro.x", "float")
        logconf.add_variable("gyro.y", "float")
        logconf.add_variable("gyro.z", "float")
    if group == "target_pos":
        logconf.add_variable("ctrltarget.x", "float")
        logconf.add_variable("ctrltarget.y", "float")
        logconf.add_variable("ctrltarget.z", "float")
    if group == "target_vel":
        logconf.add_variable("ctrltarget.vx", "float")
        logconf.add_variable("ctrltarget.vy", "float")
        logconf.add_variable("ctrltarget.vz", "float")
    # if group == "target_attitude":
    #    logconf.add_variable('ctrltarget.roll', 'float')
    #    logconf.add_variable('ctrltarget.pitch', 'float')
    #    logconf.add_variable('ctrltarget.yaw', 'float')
    if group == "controller_cmd":
        logconf.add_variable("controller.cmd_thrust", "float")
        logconf.add_variable("controller.cmd_roll", "float")
        logconf.add_variable("controller.cmd_pitch", "float")
        logconf.add_variable("controller.cmd_yaw", "float")
    if group == "controller_attitude":
        logconf.add_variable("controller.roll", "float")
        logconf.add_variable("controller.pitch", "float")
        logconf.add_variable("controller.yaw", "float")
    if group == "controller_attitude_rate":
        logconf.add_variable("controller.rollRate", "float")
        logconf.add_variable("controller.pitchRate", "float")
        logconf.add_variable("controller.yawRate", "float")
    conf_list.append(logconf)

# Prepare for liftoff
with QualisysCrazyflie(
    cf_body_name, cf_uri, world, marker_ids=cf_marker_ids, qtm_ip=qtm_ip
) as qcf:

    # Let there be time
    t = time()
    last_saved_t = time()
    save_freq = sampling_rate
    dt = 0

    ## PID Controllers ##############################################################
    # The following controllers are ordered from the low-level to high-level
    # TODO: To set the PID gains, use cfclient (see link below) instead of using the code
    # https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/tuning_tab/
    # 1. Set PID attitude rate gains
    """
    # roll
    qcf.cf.param.set_value('pid_rate.roll_kp', 10.0) #default: ?
    qcf.cf.param.set_value('pid_rate.roll_ki', 0.0) #default: ?
    qcf.cf.param.set_value('pid_rate.roll_kd', 0.2) #default: ?
    # pitch
    qcf.cf.param.set_value('pid_rate.pitch_kp', 13.0) #default ?
    qcf.cf.param.set_value('pid_rate.pitch_ki', 0.0) # default: ?
    qcf.cf.param.set_value('pid_rate.pitch_kd', 1.0) #default: ?
    # yaw
    qcf.cf.param.set_value('pid_rate.yaw_kp', 8.0) #default: ?
    qcf.cf.param.set_value('pid_rate.yaw_ki', 0.0) #default: ?
    qcf.cf.param.set_value('pid_rate.yaw_kd', 0.35) #default: ?
    """
    # 2. Set PID attitude gains
    """
    # roll
    qcf.cf.param.set_value('pid_attitude.roll_kp', 10.0) #default: 10
    qcf.cf.param.set_value('pid_attitude.roll_ki', 0.0) #default: 0.0
    qcf.cf.param.set_value('pid_attitude.roll_kd', 0.2) #default: 0.2
    # pitch
    qcf.cf.param.set_value('pid_attitude.pitch_kp', 13.0) #default 13.0
    qcf.cf.param.set_value('pid_attitude.pitch_ki', 0.0) # default: 0.0
    qcf.cf.param.set_value('pid_attitude.pitch_kd', 1.0) #default: 1.0
    # yaw
    qcf.cf.param.set_value('pid_attitude.yaw_kp', 8.0) #default: 8.0
    qcf.cf.param.set_value('pid_attitude.yaw_ki', 0.0) #default: 0.0
    qcf.cf.param.set_value('pid_attitude.yaw_kd', 0.35) #default: 0.35
    """
    # 3. Set PID velocity gains
    """
    # vx
    qcf.cf.param.set_value('velCtlPid.vxKp', 10.0) #default: ?
    qcf.cf.param.set_value('velCtlPid.vxKi', 0.0) #default: ?
    qcf.cf.param.set_value('velCtlPid.vxKd', 0.2) #default: ?
    # vy
    qcf.cf.param.set_value('velCtlPid.vyKp', 13.0) #default ?
    qcf.cf.param.set_value('velCtlPid.vyKi', 0.0) # default: ?
    qcf.cf.param.set_value('velCtlPid.vyKd', 1.0) #default: ?
    # vz
    qcf.cf.param.set_value('velCtlPid.vzKp', 8.0) #default: ?
    qcf.cf.param.set_value('velCtlPid.vzKi', 0.0) #default: ?
    qcf.cf.param.set_value('velCtlPid.vzKd', 0.35) #default: ?
    """
    # 4. Set PID position gains
    """
    # x
    qcf.cf.param.set_value('posCtlPid.xKp', 10.0) #default: ?
    qcf.cf.param.set_value('posCtlPid.xKi', 0.0) #default: ?
    qcf.cf.param.set_value('posCtlPid.xKd', 0.2) #default: ?
    # y
    qcf.cf.param.set_value('posCtlPid.yKp', 13.0) #default ?
    qcf.cf.param.set_value('posCtlPid.yKi', 0.0) # default: ?
    qcf.cf.param.set_value('posCtlPid.yKd', 1.0) #default: ?
    # z
    qcf.cf.param.set_value('posCtlPid.zKp', 8.0) #default: ?
    qcf.cf.param.set_value('posCtlPid.zKi', 0.0) #default: ?
    qcf.cf.param.set_value('posCtlPid.zKd', 0.35) #default: ?
    """
    # Log PID gains
    pid_gains = {
        "pid_rate": {
            "roll_kp": qcf.cf.param.get_value("pid_rate.roll_kp"),
            "roll_ki": qcf.cf.param.get_value("pid_rate.roll_ki"),
            "roll_kd": qcf.cf.param.get_value("pid_rate.roll_kd"),
            "pitch_kp": qcf.cf.param.get_value("pid_rate.pitch_kp"),
            "pitch_ki": qcf.cf.param.get_value("pid_rate.pitch_ki"),
            "pitch_kd": qcf.cf.param.get_value("pid_rate.pitch_kd"),
            "yaw_kp": qcf.cf.param.get_value("pid_rate.yaw_kp"),
            "yaw_ki": qcf.cf.param.get_value("pid_rate.yaw_ki"),
            "yaw_kd": qcf.cf.param.get_value("pid_rate.yaw_kd"),
        },
        "pid_attitude": {
            "roll_kp": qcf.cf.param.get_value("pid_attitude.roll_kp"),
            "roll_ki": qcf.cf.param.get_value("pid_attitude.roll_ki"),
            "roll_kd": qcf.cf.param.get_value("pid_attitude.roll_kd"),
            "pitch_kp": qcf.cf.param.get_value("pid_attitude.pitch_kp"),
            "pitch_ki": qcf.cf.param.get_value("pid_attitude.pitch_ki"),
            "pitch_kd": qcf.cf.param.get_value("pid_attitude.pitch_kd"),
            "yaw_kp": qcf.cf.param.get_value("pid_attitude.yaw_kp"),
            "yaw_ki": qcf.cf.param.get_value("pid_attitude.yaw_ki"),
            "yaw_kd": qcf.cf.param.get_value("pid_attitude.yaw_kd"),
        },
        "velCtlPid": {
            "vxKp": qcf.cf.param.get_value("velCtlPid.vxKp"),
            "vxKi": qcf.cf.param.get_value("velCtlPid.vxKi"),
            "vxKd": qcf.cf.param.get_value("velCtlPid.vxKd"),
            "vyKp": qcf.cf.param.get_value("velCtlPid.vyKp"),
            "vyKi": qcf.cf.param.get_value("velCtlPid.vyKi"),
            "vyKd": qcf.cf.param.get_value("velCtlPid.vyKd"),
            "vzKp": qcf.cf.param.get_value("velCtlPid.vzKp"),
            "vzKi": qcf.cf.param.get_value("velCtlPid.vzKi"),
            "vzKd": qcf.cf.param.get_value("velCtlPid.vzKd"),
        },
        "posCtlPid": {
            "xKp": qcf.cf.param.get_value("posCtlPid.xKp"),
            "xKi": qcf.cf.param.get_value("posCtlPid.xKi"),
            "xKd": qcf.cf.param.get_value("posCtlPid.xKd"),
            "yKp": qcf.cf.param.get_value("posCtlPid.yKp"),
            "yKi": qcf.cf.param.get_value("posCtlPid.yKi"),
            "yKd": qcf.cf.param.get_value("posCtlPid.yKd"),
            "zKp": qcf.cf.param.get_value("posCtlPid.zKp"),
            "zKi": qcf.cf.param.get_value("posCtlPid.zKi"),
            "zKd": qcf.cf.param.get_value("posCtlPid.zKd"),
        },
    }
    ############################################################################

    print("Beginning maneuvers...")
    data = {}
    data["radius"] = circle_radius
    data["angular_speed"] = circle_speed_factor
    data["save_freq"] = save_freq
    data["pid_gains"] = pid_gains
    data["pose"] = []
    data["time"] = []
    data["control"] = []

    # Asyncronous logging from the flapper firmware
    for group in group_list:
        data[group] = []
    data["time"] = []
    for logconf in conf_list:
        qcf.cf.log.add_config(logconf)
    for group, logconf in zip(group_list, conf_list):
        callback = partial(log_callback, data_log=data, key=group)
        logconf.data_received_cb.add_callback(callback)
        logconf.start()

    # MAIN LOOP WITH SAFETY CHECK
    while qcf.is_safe():

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break
        print(last_key_pressed)
        # Mind the clock
        dt = time() - t

        # Calculate Crazyflie's angular position [deg] in circle, based on time
        phi = circle_speed_factor * dt

        # Take off and hover in the center of safe airspace for 5 seconds
        if dt < 5:
            print(f"[t={dt}] Maneuvering - Center...")
            # Set target
            target = Pose(world.origin.x, world.origin.y, 1.0)
            # Engage
            qcf.safe_position_setpoint(target)

        elif dt < 65:
            if circle_axis == "Z":
                print(f"[t={int(dt)}] Maneuvering - Circle around Z")
                # Set target
                _x, _y = utils.pol2cart(circle_radius, phi)
                target = Pose(
                    world.origin.x + _x, world.origin.y + _y, 1.0, yaw=90 + phi
                )
                # Engage
                qcf.safe_position_setpoint(target)
            elif circle_axis == "Y":
                print(f"[t={int(dt)}] Maneuvering - Circle around Y")
                # Set target
                _x, _z = utils.pol2cart(circle_radius, phi)
                target = Pose(world.origin.x + _x, world.origin.y, 0.95 + _z, yaw=0)
                # Engage
                qcf.safe_position_setpoint(target)
            elif circle_axis == "X":
                print(f"[t={int(dt)}] Maneuvering - Circle around X")
                # Set target
                _y, _z = utils.pol2cart(circle_radius, phi)
                target = Pose(world.origin.x, world.origin.y + _y, 0.8 + _z, yaw=0)
                # Engage
                qcf.safe_position_setpoint(target)
            elif circle_axis == "XYZ":
                print(f"[t={int(dt)}] Maneuvering - Circle around XYZ")
                # Set target
                _, _z = utils.pol2cart(0.3, 4 * phi)
                _x, _y = utils.pol2cart(circle_radius, phi)
                target = Pose(
                    world.origin.x + _x, world.origin.y + _y, 0.8 + _z, yaw=90 + phi
                )
                # Engage
                qcf.safe_position_setpoint(target)

        # Back to center
        elif dt < 70:
            print(f"[t={dt}] Maneuvering - Center...")
            # Set target
            target = Pose(world.origin.x, world.origin.y, 1.0)
            # Engage
            qcf.safe_position_setpoint(target)

        else:
            break

        if time() - last_saved_t > save_freq:
            data["pose"].append(
                (
                    qcf.pose.x,
                    qcf.pose.y,
                    qcf.pose.z,
                    qcf.pose.roll,
                    qcf.pose.pitch,
                    qcf.pose.yaw,
                    qcf.pose.rotmatrix,
                )
            )
            data["time"].append(time())
            data["control"].append((target.x, target.y, target.z))
            last_saved_t = time()

        if not qcf.is_safe():
            print("not safe")

    # Stop logging data from the flapper firmware
    for logconf in conf_list:
        logconf.stop()

    # Land
    first_z = qcf.pose.z
    landing_time = 5
    start_time = time()
    while qcf.pose.z > 0.40:
        print(qcf.pose.z)
        print("landing...")
        cur_time = time()
        target = Pose(
            world.origin.x,
            world.origin.y,
            max(-0.20, first_z * (1 - (cur_time - start_time) / landing_time)),
        )
        qcf.safe_position_setpoint(target)

# Get the current time in seconds since the epoch
current_time = time()

# Convert the time to a human-readable format
formatted_time = datetime.fromtimestamp(current_time).strftime("%Y%m%d%H%M%S")
with open(
    "circular_traj/circular_" + circle_axis + "_" + formatted_time + ".json", "w"
) as file:
    json.dump(data, file, indent=4)
