# dora-openarm-observer

A [Dora](https://dora-rs.ai/) node that collects the last observation for OpenArm.

## Usage

Connect `arm_right` / `arm_left` to the arm's `state` output: a length-one
Arrow struct containing `qpos: list<float32>`. Additional state fields are
ignored; flat position arrays are not accepted. Camera inputs contain JPEG
bytes, decoded in parallel and emitted as flattened RGB lists.

Connect `command` and each configured `arm_<side>_status` as well as `tick`.
After `start`, output waits for all configured arms to report `started` or
`aligned` for the current attempt (when `episode_attempt_id` is supplied),
then waits for fresh arm and camera samples. `stop`, `intervene`, and `quit`
pause output and clear history. Startup samples are not reused.

## Observation History

By default, `observation` contains one current row. To include an earlier row:

```yaml
args: "--policy-history-hz 30 --policy-history-delta-indices=-32,0"
```

This selects the nearest available snapshot about 32 frames earlier, followed
by the current snapshot. Offsets must be non-positive. Missing startup history
uses the earliest snapshot from the current episode; output rows follow offset
order. The defaults are `30.0` Hz and `0`, also configurable through
`POLICY_HISTORY_HZ` and `POLICY_HISTORY_DELTA_INDICES`.

Metadata carries `episode_number`, optional `episode_attempt_id`, `history_hz`,
and comma-separated `history_delta_indices` / `history_timestamps`. Timestamps
are observer snapshot times in Unix nanoseconds, not sensor capture times.
Each output row retains its snapshot's `id`; padded rows can repeat IDs/times.
Position-only consumers use the output `position` field (right arm, then left).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
