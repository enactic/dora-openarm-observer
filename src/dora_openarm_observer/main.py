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
from concurrent.futures import ThreadPoolExecutor
import cv2
import dora
import os
import pyarrow as pa
import time


def _reset_observation(observation, arms):
    """Initialize/reset observations to None and ID to 0."""
    if "right" in arms:
        observation["arm_right"] = None
        observation["camera_wrist_right"] = None
    if "left" in arms:
        observation["arm_left"] = None
        observation["camera_wrist_left"] = None
    observation["camera_head_left"] = None
    observation["camera_head_right"] = None
    observation["camera_ceiling"] = None
    observation["id"] = 0


def _decode_camera(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode JPEG camera observation")
    return image


def _build_output(
    observation, phase_classifier_result, task_prompt, metadata, decode_pool
):
    """Convert observation to Apache Arrow data and fill metadata.

    observation keys (all values are dora events with a "value" field):
      "arm_right"          – pa.array float32, len 8 (7 joints + 1 gripper)
      "arm_left"           – pa.array float32, len 8
      "camera_wrist_right" – JPEG-encoded uint8 flat array, 960×600
      "camera_wrist_left"  – JPEG-encoded uint8 flat array, 960×600
      "camera_head_left"   – JPEG-encoded uint8 flat array, 1280×720
      "camera_head_right"  – JPEG-encoded uint8 flat array, 1280×720
      "camera_ceiling"     – JPEG-encoded uint8 flat array, 960×600
      "id"                 – int64, incremented for each observation

    Output pa.StructArray fields:
      "position"           – concatenated arm positions, list<float32>
      "camera_wrist_right" – decoded RGB flat array, list<uint8>
      "camera_wrist_left"  – decoded RGB flat array, list<uint8>
      "camera_head_left"   – decoded RGB flat array, list<uint8>
      "camera_head_right"  – decoded RGB flat array, list<uint8>
      "camera_ceiling"     – decoded RGB flat array, list<uint8>
      "phase_classifier_result" – StructArray or null
      "task_prompt"        – string (language instruction for the policy)
      "id"                 – int64, incremented for each observation

    metadata is mutated to add per-camera height/width/encoding keys.
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

    camera_names = []
    if "camera_wrist_right" in observation:
        camera_names.append("camera_wrist_right")
    if "camera_wrist_left" in observation:
        camera_names.append("camera_wrist_left")
    camera_names.extend(["camera_head_left", "camera_head_right", "camera_ceiling"])
    decode_futures = {
        name: decode_pool.submit(
            _decode_camera,
            observation[name]["value"].to_numpy(),
        )
        for name in camera_names
    }

    def add_camera_observation(name):
        image = decode_futures[name].result()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)

    for name in camera_names:
        add_camera_observation(name)
    if phase_classifier_result is None:
        arrays.append(pa.array([None]))
    else:
        arrays.append(phase_classifier_result)
    names.append("phase_classifier_result")
    arrays.append(pa.array([task_prompt], type=pa.string()))
    names.append("task_prompt")
    arrays.append(pa.array([observation["id"]], type=pa.int64()))
    names.append("id")
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
    parser.add_argument(
        "--decode-workers",
        default=int(os.getenv("OBSERVER_DECODE_WORKERS", "4")),
        type=int,
        help="Number of persistent JPEG decode workers (default: 4)",
    )
    args = parser.parse_args()
    if args.decode_workers <= 0:
        raise ValueError("--decode-workers must be positive")

    arms = args.arms.split(",")
    node = dora.Node()
    observation = {}
    _reset_observation(observation, arms)
    cv2.setNumThreads(1)
    decode_pool = ThreadPoolExecutor(
        max_workers=args.decode_workers,
        thread_name_prefix="observer-jpeg",
    )
    episode_number = 0
    last_phase_classifier_result = None
    last_task_prompt = None
    last_arm_right_status = None
    last_arm_left_status = None
    command_status = "stopped"
    try:
        for event in node:
            if event["type"] != "INPUT":
                continue

            # Main process
            event_id = event["id"]
            if event_id == "tick":
                if any(v is None for v in observation.values()):
                    # If any observation isn't ready yet, we skip this tick.
                    continue
                if (
                    ("right" in arms and last_arm_right_status == "stopped")
                    or ("left" in arms and last_arm_left_status == "stopped")
                    or command_status == "stopped"
                ):
                    _reset_observation(observation, arms)
                    continue
                metadata = {
                    "episode_number": episode_number,
                    "timestamp": time.time_ns(),
                }
                arrow_observation = _build_output(
                    observation,
                    last_phase_classifier_result,
                    last_task_prompt,
                    metadata,
                    decode_pool,
                )
                node.send_output(
                    "observation",
                    arrow_observation,
                    metadata,
                )
                observation["id"] += 1
            elif event_id == "command":
                # started, stopped, aligned
                command_status = event["value"][0].as_py()
            elif event_id == "arm_right_status":
                last_arm_right_status = event["value"][0].as_py()
            elif event_id == "arm_left_status":
                last_arm_left_status = event["value"][0].as_py()
            elif event_id == "phase_classifier_result":
                last_phase_classifier_result = event["value"]
            elif event_id == "task_prompt":
                last_task_prompt = event["value"][0].as_py()
            else:
                observation[event_id] = event
    finally:
        decode_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
