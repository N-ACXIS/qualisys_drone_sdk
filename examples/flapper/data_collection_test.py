"""
qfly | Qualisys Drone SDK Example Script: Solo Crazyflie

Takes off, flies circles around Z, Y, X axes.
ESC to land at any time.
"""

import json
from functools import partial
from time import sleep, time

import pynput
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

from qfly import Pose, QualisysCrazyflie, World, utils

# SETTINGS
cf_body_name = "flapper"  # QTM rigid body name
cf_uri = "radio://0/80/2M/E7E7E7E7E7"  # Crazyflie address
cf_marker_ids = [1, 2, 3, 4]  # Active marker IDs
circle_radius = 0.5  # Radius of the circular flight path
circle_speed_factor = 0.12  # How fast the Crazyflie should move along circle
qtm_ip = "128.174.245.190"
sampling_rate = 0.1  # Hz
last_saved_t = time()
save_freq = 0.1
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


# position, velocity, time, control
data = {}
# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()


# Set up world - the World object comes with sane defaults
world = World()

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
    dt = 0

    # Get and print the PID gains
    # TODO: reset the firmware and print the default PID gains
    print("PID attitude rate roll Kp", qcf.cf.param.get_value("pid_rate.roll_kp"))
    print("PID attitude rate roll Ki", qcf.cf.param.get_value("pid_rate.roll_ki"))
    print("PID attitude rate roll Kd", qcf.cf.param.get_value("pid_rate.roll_kd"))
    print("PID attitude rate pitch Kp", qcf.cf.param.get_value("pid_rate.pitch_kp"))
    print("PID attitude rate pitch Ki", qcf.cf.param.get_value("pid_rate.pitch_ki"))
    print("PID attitude rate pitch Kd", qcf.cf.param.get_value("pid_rate.pitch_kd"))
    print("PID attitude rate yaw Kp", qcf.cf.param.get_value("pid_rate.yaw_kp"))
    print("PID attitude rate yaw Ki", qcf.cf.param.get_value("pid_rate.yaw_ki"))
    print("PID attitude rate yaw Kd", qcf.cf.param.get_value("pid_rate.yaw_kd"))
    #
    print("PID attitude roll Kp", qcf.cf.param.get_value("pid_attitude.roll_kp"))
    print("PID attitude roll Ki", qcf.cf.param.get_value("pid_attitude.roll_ki"))
    print("PID attitude roll Kd", qcf.cf.param.get_value("pid_attitude.roll_kd"))
    print("PID attitude pitch Kp", qcf.cf.param.get_value("pid_attitude.pitch_kp"))
    print("PID attitude pitch Ki", qcf.cf.param.get_value("pid_attitude.pitch_ki"))
    print("PID attitude pitch Kd", qcf.cf.param.get_value("pid_attitude.pitch_kd"))
    print("PID attitude yaw Kp", qcf.cf.param.get_value("pid_attitude.yaw_kp"))
    print("PID attitude yaw Ki", qcf.cf.param.get_value("pid_attitude.yaw_ki"))
    print("PID attitude yaw Kd", qcf.cf.param.get_value("pid_attitude.yaw_kd"))
    #
    print("PID velocity vx Kp", qcf.cf.param.get_value("velCtlPid.vxKp"))
    print("PID velocity vx Ki", qcf.cf.param.get_value("velCtlPid.vxKi"))
    print("PID velocity vx Kd", qcf.cf.param.get_value("velCtlPid.vxKd"))
    print("PID velocity vy Kp", qcf.cf.param.get_value("velCtlPid.vyKp"))
    print("PID velocity vy Ki", qcf.cf.param.get_value("velCtlPid.vyKi"))
    print("PID velocity vy Kd", qcf.cf.param.get_value("velCtlPid.vyKd"))
    print("PID velocity vz Kp", qcf.cf.param.get_value("velCtlPid.vzKp"))
    print("PID velocity vz Ki", qcf.cf.param.get_value("velCtlPid.vzKi"))
    print("PID velocity vz Kd", qcf.cf.param.get_value("velCtlPid.vzKd"))
    #
    print("PID position x Kp", qcf.cf.param.get_value("posCtlPid.xKp"))
    print("PID position x Ki", qcf.cf.param.get_value("posCtlPid.xKi"))
    print("PID position x Kd", qcf.cf.param.get_value("posCtlPid.xKd"))
    print("PID position y Kp", qcf.cf.param.get_value("posCtlPid.yKp"))
    print("PID position y Ki", qcf.cf.param.get_value("posCtlPid.yKi"))
    print("PID position y Kd", qcf.cf.param.get_value("posCtlPid.yKd"))
    print("PID position z Kp", qcf.cf.param.get_value("posCtlPid.zKp"))
    print("PID position z Ki", qcf.cf.param.get_value("posCtlPid.zKi"))
    print("PID position z Kd", qcf.cf.param.get_value("posCtlPid.zKd"))
    #
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
    data["pid_gains"] = pid_gains

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
        # Mind the clock
        dt = time() - t
        # Calculate Crazyflie's angular position in circle, based on time
        phi = circle_speed_factor * dt * 360
        if dt < 3:
            if time() - last_saved_t > save_freq:
                data["time"].append(time())
                last_saved_t = time()
                continue
        else:
            for logconf in conf_list:
                logconf.stop()
            break

# Open a file in write mode and use json.dump() to write the dictionary to the file
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)

print("Dictionary has been saved to data.json")
