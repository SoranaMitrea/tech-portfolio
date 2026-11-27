"""Tests for the safety checks before a navigation goal (C++, ziel_pruefung.hpp from slam_ziel.cpp).

A goal is only released if the pose is fresh, the goal is close enough, plausible, and the robot walks.
The tests compile the C++ code and call it through ziel_cli.
"""
import json
import pytest

JETZT = 1_800_000_000.0


def pose(x=0.0, y=0.0, alter_s=0.5, zeit=None):
    return json.dumps({'x': x, 'y': y, 'z': 0.06, 'yaw_grad': 0.0,
                       'zeit': JETZT - alter_s if zeit is None else zeit})


# ---- REQ-06: goal only with a fresh pose (at most 3 s old)

@pytest.mark.req('REQ-06')
@pytest.mark.parametrize('alter_s,erwartet', [(0.0, 0), (2.9, 0), (3.0, 0), (3.1, 5), (49.0, 5)])
def test_pose_age_limit(ziel_cli, alter_s, erwartet):
    code, _ = ziel_cli(1.0, 0.0, pose(alter_s=alter_s), JETZT, 500)
    assert code == erwartet


@pytest.mark.req('REQ-06')
@pytest.mark.parametrize('text', ['', '{}', '{"x": 1, "y": 2, "yaw_grad": 0}', 'kaputt'])
def test_missing_or_incomplete_pose_rejected(ziel_cli, text):
    assert ziel_cli(1.0, 0.0, text, JETZT, 500) == (5, 'ABGELEHNT no_fresh_pose')


@pytest.mark.req('REQ-06')
@pytest.mark.xfail(strict=True, reason='FINDING F-01: a pose with a timestamp in the future has a negative '
                                       'age and is accepted as fresh')
def test_pose_from_future_rejected(ziel_cli):
    code, _ = ziel_cli(1.0, 0.0, pose(zeit=JETZT + 3600), JETZT, 500)
    assert code == 5


# ---- REQ-07: distance to goal within the limit (default 3 m, never above 10 m)

@pytest.mark.req('REQ-07')
@pytest.mark.parametrize('zx,zy,max_m,erwartet', [
    (2.46, 0.0, 3.0, 0),     # "at the door" from the deployment case
    (3.0, 0.0, 3.0, 0),      # exactly at the limit
    (3.01, 0.0, 3.0, 5),
    (3.0, 4.0, 3.0, 5),      # 5 m diagonal
    (9.0, 0.0, 50.0, 0),     # limit is capped at 10 m ...
    (10.5, 0.0, 50.0, 5),    # ... so 10.5 m is rejected even with --max 50
])
def test_distance_limit(ziel_cli, zx, zy, max_m, erwartet):
    code, _ = ziel_cli(zx, zy, pose(), JETZT, 500, max_m)
    assert code == erwartet


@pytest.mark.req('REQ-07')
def test_distance_measured_from_current_pose(ziel_cli):
    # robot at (2, 2), goal (4, 2): 2 m away, released although 4.5 m from the map origin
    assert ziel_cli(4.0, 2.0, pose(x=2.0, y=2.0), JETZT, 500)[0] == 0


@pytest.mark.req('REQ-07')
@pytest.mark.parametrize('max_m', [0.0, -1.0])
def test_non_positive_limit_rejected(ziel_cli, max_m):
    assert ziel_cli(0.5, 0.0, pose(), JETZT, 500, max_m) == (5, 'ABGELEHNT max_not_positive')


# ---- REQ-08: goal only in walking state (fsm_id 500, 501, 801)

@pytest.mark.req('REQ-08')
@pytest.mark.parametrize('fsm,erwartet', [(500, 0), (501, 0), (801, 0), (0, 5), (1, 5), (4, 5), (200, 5), (-1, 5)])
def test_walking_state_required(ziel_cli, fsm, erwartet):
    code, _ = ziel_cli(1.0, 0.0, pose(), JETZT, fsm)
    assert code == erwartet


# ---- REQ-09: implausible goals rejected

@pytest.mark.req('REQ-09')
def test_goal_far_from_origin_rejected(ziel_cli):
    # without pose check (--ohne-pose) the plausibility check is the last protection
    assert ziel_cli(5000.0, 0.0, '', JETZT, 500, 3.0, '--ohne-pose') == (5, 'ABGELEHNT implausible')


@pytest.mark.req('REQ-09')
@pytest.mark.xfail(strict=True, reason='FINDING F-02: goal "nan nan" passes all checks, because every '
                                       'comparison with NaN is false')
def test_nan_goal_rejected(ziel_cli):
    code, _ = ziel_cli('nan', 'nan', pose(), JETZT, 500)
    assert code == 5
