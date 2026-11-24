# dora-openarm-observer

A [Dora](https://dora-rs.ai/) node that collects the last observation for OpenArm.

## Usage

The node sends `observation` that has the last data of all used arms and
cameras for each `tick`. It sends nothing until all of the following are
satisfied:

- It received `start` by `command`. `stop` by `command` stops sending again.
- It received data from all used arms and cameras.
- No used arm reports `stopped` by `arm_right_status`/`arm_left_status`.

The used cameras are the connected `camera_*` inputs: `camera_wrist_right`,
`camera_wrist_left`, `camera_head_right`, `camera_head_left` and
`camera_ceiling`. Don't connect a camera that never sends data because the
node waits for it forever. A wrist camera is used only when its arm is also
used. You can choose the used arms by `--arms` (`ARMS`): `right,left`
(default), `right` or `left`.

For example, OpenArm on a pedestal doesn't have the ceiling camera, so
`camera_ceiling` isn't connected:

```yaml
- id: observer
  path: dora-openarm-observer
  inputs:
    tick: dora/timer/millis/33
    arm_right: arm-right/position
    arm_left: arm-left/position
    camera_wrist_right: camera-wrist-right/image
    camera_wrist_left: camera-wrist-left/image
    camera_head_right: camera-head-stereo-splitter/image_1
    camera_head_left: camera-head-stereo-splitter/image_0
    command: evaluation-ui/arm_command
    arm_right_status: arm-right/status
    arm_left_status: arm-left/status
    task_prompt: evaluation-ui/task_prompt
  outputs:
    - observation
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
