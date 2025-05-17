from typing import List

import requests

from api.schema import TargetRequest, TargetResponse
from qfly import Pose

# Configure your server URL
SERVER_URL = "http://localhost:8000"


def get_target_position(req: TargetRequest) -> Pose:
    resp = requests.post(f"{SERVER_URL}/compute_target", json=req.model_dump())
    resp.raise_for_status()
    data = TargetResponse(**resp.json())
    return Pose(data.x, data.y, data.z, yaw=data.yaw), data.status
