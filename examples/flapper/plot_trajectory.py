import argparse
import json
import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Load JSON file
with open("examples/flapper/circular_traj/circular_Z_20250423183719.json", "r") as f:
    data = json.load(f)

# Plot positions
poses = data["pose"]
x_qual = np.array([p[0] for p in poses])
y_qual = np.array([p[1] for p in poses])
z_qual = np.array([p[2] for p in poses])

targets = data["control"]
x_target = np.array([p[0] for p in targets])
y_target = np.array([p[1] for p in targets])
z_target = np.array([p[2] for p in targets])

target_pos = data["target_pos"]
x_ctrltarget = np.array([c["ctrltarget.x"] for c in target_pos])
y_ctrltarget = np.array([c["ctrltarget.y"] for c in target_pos])
z_ctrltarget = np.array([c["ctrltarget.z"] for c in target_pos])

position = data["pos"]  # [m]
x_state = np.array([p["stateEstimate.x"] for p in position])
y_state = np.array([p["stateEstimate.y"] for p in position])
z_state = np.array([p["stateEstimate.z"] for p in position])

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(x_qual, label="x qualisys")
axs[0].plot(x_state, label="x state est")
axs[0].plot(x_target, label="x target")
axs[0].plot(x_ctrltarget, label="ctrltarget.x")
axs[0].legend()
axs[1].plot(y_qual, label="y qualisys")
axs[1].plot(y_state, label="y state est")
axs[1].plot(y_target, label="y target")
axs[1].plot(y_ctrltarget, label="ctrltarget.y")
axs[1].legend()
axs[2].plot(z_qual, label="z qualisys")
axs[2].plot(z_state, label="z state est")
axs[2].plot(z_target, label="z target")
axs[2].plot(z_ctrltarget, label="ctrltarget.z")
axs[2].legend()
plt.show()

# Plot velocity
velocity = data["vel"]  # [m/s]
vx_state = np.array([v["stateEstimate.vx"] for v in velocity])
vy_state = np.array([v["stateEstimate.vy"] for v in velocity])
vz_state = np.array([v["stateEstimate.vz"] for v in velocity])

dt = 0.1
vx_deriv = np.gradient(x_state, dt)
vy_deriv = np.gradient(y_state, dt)
vz_deriv = np.gradient(z_state, dt)

target_vel = data["target_vel"]
vx_ctrltarget = np.array([c["ctrltarget.vx"] for c in target_vel])
vy_ctrltarget = np.array([c["ctrltarget.vy"] for c in target_vel])
vz_ctrltarget = np.array([c["ctrltarget.vz"] for c in target_vel])

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(vx_state, label="vx state est")
axs[0].plot(vx_ctrltarget, label="ctrltarget.vx")
axs[0].plot(vx_deriv, label="vx by deriv")
axs[0].legend()
axs[1].plot(vy_state, label="vy state est")
axs[1].plot(vy_ctrltarget, label="ctrltarget.vy")
axs[1].plot(vy_deriv, label="vy by deriv")
axs[1].legend()
axs[2].plot(vz_state, label="vz state est")
axs[2].plot(vz_ctrltarget, label="ctrltarget.vz")
axs[2].plot(vz_deriv, label="vz by deriv")
axs[2].legend()
plt.show()

# Plot acceleration
acc = data["acc"]  # [g]
ax_state = np.array([a["stateEstimate.ax"] for a in acc]) * 9.81  # [m/s/s]
ay_state = np.array([a["stateEstimate.ay"] for a in acc]) * 9.81  # [m/s/s]
az_state = np.array([a["stateEstimate.az"] for a in acc]) * 9.81  # [m/s/s]

dt = 0.1
ax_deriv = np.gradient(vx_state, dt)
ay_deriv = np.gradient(vy_state, dt)
az_deriv = np.gradient(vz_state, dt)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(ax_state, label="ax state est")
axs[0].plot(ax_deriv, label="ax by deriv")
axs[0].legend()
axs[1].plot(ay_state, label="ay state est")
axs[1].plot(ay_deriv, label="ay by deriv")
axs[1].legend()
axs[2].plot(az_state, label="az state est")
axs[2].plot(az_deriv, label="az by deriv")
axs[2].legend()
plt.show()

# Plot controller commands
controller_cmd = data["controller_cmd"]  # [deg]?
cmd_thrust = np.array([c["controller.cmd_thrust"] for c in controller_cmd])
cmd_roll = (
    np.array([c["controller.cmd_roll"] for c in controller_cmd]) / 180 * math.pi
)  # [rad]?
cmd_pitch = (
    np.array([c["controller.cmd_pitch"] for c in controller_cmd]) / 180 * math.pi
)  # [rad]?
cmd_yaw = (
    np.array([c["controller.cmd_yaw"] for c in controller_cmd]) / 180 * math.pi
)  # [rad]?

# Plot attitude
stabilizer = data["stabilizer"]  # [deg]
state_roll = (
    np.array([c["stabilizer.roll"] for c in stabilizer]) / 180 * math.pi
)  # [rad]
state_pitch = (
    np.array([c["stabilizer.pitch"] for c in stabilizer]) / 180 * math.pi
)  # [rad]
state_yaw = np.array([c["stabilizer.yaw"] for c in stabilizer]) / 180 * math.pi  # [rad]

controller_attitude = data["controller_attitude"]  # [deg]
controller_roll = (
    np.array([c["controller.roll"] for c in controller_attitude]) / 180 * math.pi
)  # [rad]
controller_pitch = (
    np.array([c["controller.pitch"] for c in controller_attitude]) / 180 * math.pi
)  # [rad]
controller_yaw = (
    np.array([c["controller.yaw"] for c in controller_attitude]) / 180 * math.pi
)  # [rad]

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(state_roll, label="roll state")
axs[0].plot(controller_roll, label="roll ctrl reference")
# axs[0].plot(cmd_roll, label='cmd roll')
axs[0].legend()
axs[1].plot(state_pitch, label="pitch state")
axs[1].plot(controller_pitch, label="pitch ctrl reference")
# axs[1].plot(cmd_pitch, label='cmd pitch')
axs[1].legend()
axs[2].plot(state_yaw, label="yaw state")
axs[2].plot(controller_yaw, label="yaw ctrl reference")
# axs[2].plot(cmd_yaw, label='cmd yaw')
axs[2].legend()
plt.show()

# Plot body angular velocity
# NOTE:
# stateCompressed.rateRoll = sensorData.gyro.x * deg2millirad;
# stateCompressed.ratePitch = -sensorData.gyro.y * deg2millirad;
# stateCompressed.rateYaw = sensorData.gyro.z * deg2millirad;
gyro = data["gyro"]  # [deg/s]
gyro_x = np.array([c["gyro.x"] for c in gyro]) / 180 * math.pi  # [rad/s]
gyro_y = np.array([c["gyro.y"] for c in gyro]) / 180 * math.pi  # [rad/s]
gyro_z = np.array([c["gyro.z"] for c in gyro]) / 180 * math.pi  # [rad/s]

attitude_rate = data["attitude_rate"]  # [milliradians / sec]
roll_rate = (
    np.array([c["stateEstimateZ.rateRoll"] for c in attitude_rate]) / 1000
)  # [rad/s]
pitch_rate = (
    np.array([c["stateEstimateZ.ratePitch"] for c in attitude_rate]) / 1000
)  # [rad/s]
yaw_rate = (
    np.array([c["stateEstimateZ.rateYaw"] for c in attitude_rate]) / 1000
)  # [rad/s]

controller_attitude_rate = data["controller_attitude_rate"]  # [deg/s]
controller_roll_rate = (
    np.array([c["controller.rollRate"] for c in controller_attitude_rate])
    / 180
    * math.pi
)  # [rad/s]
controller_pitch_rate = (
    np.array([c["controller.pitchRate"] for c in controller_attitude_rate])
    / 180
    * math.pi
)  # [rad/s]
controller_yaw_rate = (
    np.array([c["controller.yawRate"] for c in controller_attitude_rate])
    / 180
    * math.pi
)  # [rad/s]

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(gyro_x, label="gyro_x")
axs[0].plot(roll_rate, label="roll_rate state")
axs[0].plot(controller_roll_rate, label="roll_rate ctrl reference")
axs[0].legend()
axs[1].plot(-gyro_y, label="-gyro_y")
axs[1].plot(pitch_rate, label="pitch_rate state")
axs[1].plot(controller_pitch_rate, label="pitch_rate ctrl reference")
axs[1].legend()
axs[2].plot(gyro_z, label="gyro_z")
axs[2].plot(yaw_rate, label="yaw_rate state")
axs[2].plot(controller_yaw_rate, label="yaw_rate ctrl reference")
axs[2].legend()
plt.show()

gyro_x_dot = np.gradient(gyro_x, dt)  # [rad/s/s]
gyro_y_dot = np.gradient(gyro_y, dt)  # [rad/s/s]
gyro_z_dot = np.gradient(gyro_z, dt)  # [rad/s/s]

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(gyro_x_dot, label="gyro_x_dot")
axs[0].legend()
axs[1].plot(gyro_y_dot, label="gyro_y_dot")
axs[1].legend()
axs[2].plot(gyro_z_dot, label="gyro_z_dot")
axs[2].legend()
plt.show()

# Plot Roll_dot, pitch_dot, yaw_dot
roll_rate_deriv = np.gradient(state_roll, dt)
pitch_rate_deriv = np.gradient(state_pitch, dt)
yaw_rate_deriv = np.gradient(np.unwrap(2 * state_yaw) / 2, dt)

attitude_rate_trans = np.zeros((state_roll.shape[0], 3))
for i in range(state_roll.shape[0]):
    roll_ = state_roll[i]
    pitch_ = state_pitch[i]
    R_hat = np.array(
        [
            [1, np.sin(roll_) * np.tan(pitch_), np.cos(roll_) * np.tan(pitch_)],
            [0, np.cos(roll_), -np.sin(roll_)],
            [0, np.sin(roll_) / np.cos(pitch_), np.cos(roll_) / np.cos(pitch_)],
        ]
    )
    omega = np.array([gyro_x[i], -gyro_y[i], gyro_z[i]])
    attitude_rate_trans[i, :] = R_hat @ omega

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(roll_rate_deriv, label="roll rate deriv")
axs[0].plot(attitude_rate_trans[:, 0], label="roll rate trans")
axs[0].legend()
axs[1].plot(pitch_rate_deriv, label="pitch rate deriv")
axs[1].plot(attitude_rate_trans[:, 1], label="pitch rate trans")
axs[1].legend()
axs[2].plot(yaw_rate_deriv, label="yaw rate deriv")
axs[2].plot(attitude_rate_trans[:, 2], label="yaw rate trans")
axs[2].legend()
plt.show()
