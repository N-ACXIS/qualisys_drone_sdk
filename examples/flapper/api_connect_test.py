from api.schema import TargetRequest
from api.service import get_target_position
from qfly import World

# SETTINGS
cf_body_name = "flapper"
cf_marker_ids = [1, 2, 3, 4]
circle_radius = 0.75
circle_axis = "XYZ"
circle_speed_factor = 9
qtm_ip = "128.174.245.190"

sampling_rate = 0.1
last_key_pressed = None

world = World(expanse=1.8, speed_limit=1.1)
# preflight check
req = TargetRequest(
    dt=0,
    axis=circle_axis,
    radius=circle_radius,
    speed=circle_speed_factor,
    origin_x=world.origin.x,
    origin_y=world.origin.y,
    origin_z=world.origin.z,
    x_cur=world.origin.x,
    y_cur=world.origin.y,
    z_cur=world.origin.z,
    rot_mat=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
)
print(req.model_dump(exclude_unset=False))
target, status = get_target_position(req)
if status != "OK":
    print(f"Error: {status}")

print(target, status)
