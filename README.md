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

### Independent State History

An optional state history can use a different set of offsets from the image rows:

```yaml
args: "--policy-history-delta-indices=-24,-18,-12,-6,0 --policy-history-hz 30 --policy-state-history-delta-indices=-6,-5,-4,-3,-2,-1,0 --policy-state-history-hz 30"
```

The equivalent environment variables are `POLICY_STATE_HISTORY_DELTA_INDICES`
and `POLICY_STATE_HISTORY_HZ`; the latter defaults to `POLICY_HISTORY_HZ`.
When state offsets are omitted, the old schema and behavior are unchanged.
Independent state offsets must be unique, increasing, non-positive and end at
zero. The image offsets must contain exactly one current (`0`) row.

The example still emits **five image rows** with the same five legacy `position`
rows. An optional `state_history: list<list<float32>>` field holds the independent
seven-by-DOF matrix only on the current image row; its other rows are null. No
extra history images are serialized. Metadata adds `state_history_delta_indices`,
`state_history_hz`, and `state_history_timestamps`, all using the same snapshot
clock as image history. The shared snapshot buffer covers the longer of the two
requested time windows, and both histories clear on episode lifecycle changes.
Missing startup state samples repeat the current episode's earliest snapshot.
Consumers that need dense state history must read the explicit field instead of
assuming the legacy sparse `position` rows have changed meaning.

The `command` input accepts `quit` to end the observer and close its observation
output, including while waiting for Start or arm readiness. Dora `STOP` also ends
the loop. Both paths wait for the JPEG worker pool to finish. `stop` and
`intervene` clear the episode but keep the node available for another Start.
The tick input can therefore connect directly to a Dora timer, and consumers
can subscribe directly to `observer/observation` without a quit relay.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
