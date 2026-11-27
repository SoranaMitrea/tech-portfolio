"""Tests for the pose bridge g1_slam_pose.py (reads /slam_info, writes /tmp/g1_pose.json).

The module is imported directly: rclpy is only imported inside main(), so the logic runs without ROS.
"""
import json, os, subprocess, sys
import pytest
import g1_slam_pose as bruecke


def meldung(typ='pos_info', **pose):
    p = {'x': 1.0, 'y': -0.5, 'z': 0.06, 'q_w': 1.0, 'q_x': 0.0, 'q_y': 0.0, 'q_z': 0.0}
    p.update(pose)
    return json.dumps({'type': typ, 'data': {'currentPose': p, 'pcdName': 'TWFwMQ_20261008'}})


# ---- REQ-01: only pos_info produces a pose

@pytest.mark.req('REQ-01')
def test_pos_info_gives_pose():
    p = bruecke.pose_aus_meldung(meldung())
    assert p['x'] == 1.0 and p['y'] == -0.5 and p['karte'] == 'TWFwMQ_20261008'


@pytest.mark.req('REQ-01')
@pytest.mark.parametrize('typ', ['map_info', 'status', '', None])
def test_other_message_types_ignored(typ):
    assert bruecke.pose_aus_meldung(meldung(typ=typ)) is None


# ---- REQ-02: malformed or incomplete messages give no pose and no crash

@pytest.mark.req('REQ-02')
@pytest.mark.parametrize('text', [
    '',                                                   # empty
    '{not json',                                          # broken JSON
    '{"type": "pos_info"}',                               # no data
    '{"type": "pos_info", "data": {}}',                   # no currentPose
    '{"type": "pos_info", "data": {"currentPose": null}}',
    meldung().replace('"q_w": 1.0, ', ''),                # quaternion incomplete
    meldung(x='abc'),                                     # not a number
])
def test_malformed_message_gives_none(text):
    assert bruecke.pose_aus_meldung(text) is None


# ---- REQ-03: heading in degrees from the quaternion

@pytest.mark.req('REQ-03')
@pytest.mark.parametrize('w,z,grad', [
    (1.0, 0.0, 0.0),
    (0.7071068, 0.7071068, 90.0),
    (0.7071068, -0.7071068, -90.0),
    (0.0, 1.0, 180.0),
])
def test_yaw_from_quaternion(w, z, grad):
    assert bruecke.yaw_aus_quaternion(w, 0.0, 0.0, z) == pytest.approx(grad, abs=1e-3)


# ---- REQ-04: pose file is complete JSON with timestamp, written atomically

@pytest.mark.req('REQ-04')
def test_pose_file_complete_with_timestamp(tmp_path, monkeypatch):
    ziel = str(tmp_path / 'g1_pose.json')
    monkeypatch.setattr(bruecke, 'AUSGABE', ziel)
    bruecke.schreiben(bruecke.pose_aus_meldung(meldung()))
    d = json.load(open(ziel))
    assert {'x', 'y', 'yaw_grad', 'zeit'} <= d.keys()
    assert not os.path.exists(ziel + '.tmp')


@pytest.mark.req('REQ-04')
def test_pose_file_replaced_atomically(tmp_path, monkeypatch):
    ziel = str(tmp_path / 'g1_pose.json')
    monkeypatch.setattr(bruecke, 'AUSGABE', ziel)
    aufrufe = []
    original = os.replace
    monkeypatch.setattr(bruecke.os, 'replace', lambda a, b: (aufrufe.append((a, b)), original(a, b)))
    bruecke.schreiben({'x': 0, 'y': 0, 'yaw_grad': 0})
    assert aufrufe == [(ziel + '.tmp', ziel)]


# ---- REQ-05: only one bridge at a time

@pytest.mark.req('REQ-05')
def test_second_bridge_refused(tmp_path, monkeypatch):
    sperre = str(tmp_path / 'g1_slam_pose.lock')
    monkeypatch.setattr(bruecke, 'SPERRE', sperre)
    erste = bruecke.sperren()
    try:
        code = ('import sys; sys.path.insert(0, %r); import g1_slam_pose as b; '
                'b.SPERRE = %r; b.sperren()' % (os.path.dirname(bruecke.__file__), sperre))
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert r.returncode == 1
        assert 'ABBRUCH' in r.stderr and str(os.getpid()) in r.stderr
    finally:
        erste.close()


@pytest.mark.req('REQ-05')
def test_bridge_can_start_after_first_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(bruecke, 'SPERRE', str(tmp_path / 'g1_slam_pose.lock'))
    bruecke.sperren().close()
    bruecke.sperren().close()   # must not exit
