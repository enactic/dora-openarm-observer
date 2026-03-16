# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to collect the last observation."""

import argparse
import cv2
import dora
import os
import pyarrow as pa
import time


def _build_output(observation, phase_classifier_result, metadata):
    """Convert observation to Apache Arrow data and fill metadata.

    --arms=right,left case:

    observation: {
      # len: 8 (7 joints + 1 gripper)
      "arm_right": {"value": pa.array([...], type=pa.float32())},
      # len: ?, encoding: JPEG, width: 960, height: 600
      "camera_wrist_right": {"value": pa.array([...], type=pa.uint8())},
      # len: 8 (7 joints + 1 gripper)
      "arm_left": {"value": pa.array([...], type=pa.float32())},
      # len: ?, encoding: JPEG, width: 960, height: 600
      "camera_wrist_left": {"value": pa.array([...], type=pa.uint8())},
      # len: ?, encoding: JPEG, width: 960, height: 600
      "camera_head": {"value": pa.array([...], type=pa.uint8())},
      # len: ?, encoding: JPEG, width: 960, height: 600
      "camera_ceiling": {"value": pa.array([...], type=pa.uint8())},
    }

    phase_classifier_result: {
      "value": pa.StructArray: {
        "phase": pa.int32(),
        "phase_name": pa.string(),
        "confidence": pa.float32(),
        "success": pa.bool_(),
        "status": pa.string(),
      },
    }

    ->

    pa.StructArray: {
      # element len: 8 (7 joints + 1 gripper) * 2 (right + left)
      # "arm_right" + "arm_left"
      "position": pa.list_(pa.float32()),
      # element len: 600 (height) * 960 (width) * 3 (RGB)
      # element shape: (height, width, color)
      "camera_wrist_right": pa.list_(pa.uint8()),
      # element len: 600 (height) * 960 (width) * 3 (RGB)
      # element shape: (height, width, color)
      "camera_wrist_left": pa.list_(pa.uint8()),
      # element len: 600 (height) * 960 (width) * 3 (RGB)
      # element shape: (height, width, color)
      "camera_head": pa.list_(pa.uint8()),
      # element len: 600 (height) * 960 (width) * 3 (RGB)
      # element shape: (height, width, color)
      "camera_ceiling": pa.list_(pa.uint8()),
      # Use the given phase_classifier_result as-is
      "phase_classifier_result": pa.StructArray() or pa.null(),
    }
    """
    arrays = []
    names = []
    position_arrays = []
    if "arm_right" in observation:
        position_arrays.append(observation["arm_right"]["value"])
    if "arm_left" in observation:
        position_arrays.append(observation["arm_left"]["value"])
    arrays.append(
        pa.array(
            [pa.concat_arrays(position_arrays)], type=pa.list_(position_arrays[0].type)
        )
    )
    names.append("position")

    def add_camera_observation(name):
        camera = observation[name]
        image = cv2.imdecode(
            camera["value"].to_numpy(),
            cv2.IMREAD_UNCHANGED,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)

    if "camera_wrist_right" in observation:
        add_camera_observation("camera_wrist_right")
    if "camera_wrist_left" in observation:
        add_camera_observation("camera_wrist_left")
    add_camera_observation("camera_head")
    add_camera_observation("camera_ceiling")
    if phase_classifier_result is None:
        arrays.append(pa.array([None]))
    else:
        arrays.append(phase_classifier_result)
    names.append("phase_classifier_result")
    return pa.StructArray.from_arrays(arrays, names)


def main():
    """Collect the last observation."""
    parser = argparse.ArgumentParser(description="Collect the last observation")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    args = parser.parse_args()
    arms = args.arms.split(",")
    node = dora.Node()
    observation = {}
    if "right" in arms:
        observation["arm_right"] = None
        observation["camera_wrist_right"] = None
    if "left" in arms:
        observation["arm_left"] = None
        observation["camera_wrist_left"] = None
    observation["camera_head"] = None
    observation["camera_ceiling"] = None
    episode_number = 0
    last_phase_classifier_result = None
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        event_id = event["id"]
        if event_id == "tick":
            if any(v is None for v in observation.values()):
                # If any observation isn't ready yet, we skip this tick.
                continue
            metadata = {
                "episode_number": episode_number,
                "timestamp": time.time_ns(),
            }
            arrow_ovservation = _build_output(
                observation, last_phase_classifier_result, metadata
            )
            node.send_output(
                "observation",
                arrow_ovservation,
                metadata,
            )
        elif event_id == "command":
            command = event["value"][0].as_py()
            if command == "start":
                episode_number = event["metadata"].get("episode_number", 0)
        elif event_id == "phase_classifier_result":
            last_phase_classifier_result = event["value"]
        else:
            observation[event_id] = event


if __name__ == "__main__":
    main()
