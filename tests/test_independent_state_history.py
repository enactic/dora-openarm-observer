"""Separate state sampling must not add image rows or reuse an old episode."""
# ruff: noqa: D103

from collections import deque

import cv2
import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_observer import main as observer


def snapshot(index):
    return observer._ObservationSnapshot(
        np.full(16, index, np.float32),
        {name: np.full((2, 3, 3), index, np.uint8) for name in observer.CAMERA_ORDER},
        None, None, "test", index,
    )


def test_five_image_rows_and_seven_states_are_independent():
    history = deque((round(i * 1e9 / 30), snapshot(i)) for i in range(31))
    slow = (-24, -18, -12, -6, 0)
    fast = tuple(range(-6, 1))
    images, image_times = observer._select_history_items(history, history[-1][0], slow, 30)
    states, state_times = observer._select_history_items(history, history[-1][0], fast, 30)
    metadata = {"history_delta_indices": ",".join(map(str, slow))}
    legacy = observer._build_output(images, dict(metadata))
    split = observer._build_output(images, metadata, state_snapshots=states)
    assert len(split) == 5
    for name in legacy.type.names:
        assert split.field(name).equals(legacy.field(name))
    field = split.field("state_history")
    assert field.type == pa.list_(pa.list_(pa.float32()))
    assert field.to_pylist()[:4] == [None] * 4
    np.testing.assert_array_equal(field[-1].as_py(), [[i] * 16 for i in range(24, 31)])
    assert image_times == [round(i * 1e9 / 30) for i in (6, 12, 18, 24, 30)]
    assert state_times == [round(i * 1e9 / 30) for i in range(24, 31)]
    # The auxiliary matrix adds hundreds of bytes, not another image history.
    assert split.nbytes - legacy.nbytes < 1024


def test_auxiliary_matrix_follows_current_row_not_row_order():
    output = observer._build_output([snapshot(9), snapshot(3)],
        {"history_delta_indices": "0,-6"}, state_snapshots=[snapshot(8), snapshot(9)])
    assert output.field("state_history")[0].as_py() == [[8.] * 16, [9.] * 16]
    assert output.field("state_history")[1].as_py() is None


def test_state_window_can_be_longer_than_image_window_and_resets(monkeypatch):
    outputs, events = [], []
    qtype = pa.struct([("qpos", pa.list_(pa.float32()))])
    def add(name, value=None, metadata=None):
        events.append({"type": "INPUT", "id": name, "value": value, "metadata": metadata or {}})
    def start(attempt):
        metadata = {"episode_attempt_id": attempt}
        add("command", pa.array(["start"]), metadata)
        for side in ("right", "left"):
            add("arm_" + side + "_status", pa.array(["started"]), metadata)
    def samples(index):
        for side in ("right", "left"):
            add("arm_" + side, pa.array([{"qpos": [float(index)] * 8}], type=qtype))
        ok, jpeg = cv2.imencode(".jpg", np.full((2, 3, 3), index, np.uint8))
        assert ok
        for camera in observer.CAMERA_ORDER:
            add(camera, pa.array(jpeg.reshape(-1)))
        add("tick")
    start("first")
    for index in range(16):
        samples(index)
    add("command", pa.array(["stop"]))
    start("second")
    samples(99)
    class Node(list):
        def send_output(self, name, value, metadata):
            outputs.append((value, dict(metadata)))
    times = iter(1_000_000_000 + round(i * 1e9 / 30) for i in range(17))
    monkeypatch.setattr(observer.time, "time_ns", lambda: next(times))
    monkeypatch.setattr(observer.dora, "Node", lambda: Node(events))
    monkeypatch.setattr("sys.argv", ["observer", "--policy-history-delta-indices=0",
        "--policy-state-history-delta-indices=-12,0", "--policy-state-history-hz=30"])
    observer.main()
    assert len(outputs) == 17
    np.testing.assert_array_equal(outputs[15][0].field("state_history")[0].as_py(), [[3] * 16, [15] * 16])
    value, meta = outputs[-1]
    assert meta["episode_attempt_id"] == "second" and meta["state_history_delta_indices"] == "-12,0"
    assert meta["state_history_hz"] == 30
    assert meta["state_history_timestamps"].split(",") == [str(meta["timestamp"])] * 2
    np.testing.assert_array_equal(value.field("state_history")[0].as_py(), [[99] * 16] * 2)


@pytest.mark.parametrize("args", [
    ["--policy-state-history-delta-indices=-1,-1,0"],
    ["--policy-state-history-delta-indices=0,-1"],
    ["--policy-state-history-delta-indices=-1"],
    ["--policy-state-history-delta-indices=-1,0", "--policy-state-history-hz=nan"],
    ["--policy-state-history-delta-indices=-1,0", "--policy-history-delta-indices=-2"],
])
def test_invalid_independent_history_fails_before_node_start(monkeypatch, args):
    monkeypatch.setattr(observer.dora, "Node", lambda: pytest.fail("Node started"))
    monkeypatch.setattr("sys.argv", ["observer", *args])
    with pytest.raises(ValueError):
        observer.main()
