# Requirements – G1 goal navigation (pose bridge and goal checks)

Derived from the deployment case. Each requirement has an ID, an acceptance criterion and a verification method.
Code under test: `src/g1_slam_pose.py` (pose bridge) and `src/ziel_pruefung.hpp` (checks from `slam_ziel.cpp`).

| ID | Requirement | Acceptance criterion | Method |
|---|---|---|---|
| REQ-01 | The bridge only takes pose messages | A `pos_info` message gives a pose; all other message types give none | Test |
| REQ-02 | Malformed messages must not stop the bridge | Broken JSON, missing fields or non-numbers give no pose and no exception | Test |
| REQ-03 | The heading is given in degrees | Yaw from the quaternion: 0°, 90°, −90°, 180° within 0.001° | Test |
| REQ-04 | Other programs never read a half-written pose | Pose file is complete JSON with `zeit`; written to `.tmp` and replaced atomically | Test |
| REQ-05 | Only one pose bridge runs at a time | A second bridge exits with code 1 and names the running PID; a new start works after the first ends | Test |
| REQ-06 | A goal is only sent with a fresh pose | Pose at most 3 s old; missing, incomplete or older pose → rejected | Test |
| REQ-07 | A goal is only sent within the distance limit | Distance from current pose ≤ limit (default 3 m); limit capped at 10 m; limit ≤ 0 → rejected | Test |
| REQ-08 | A goal is only sent while the robot walks | fsm_id 500, 501 or 801 → released; any other state → rejected | Test |
| REQ-09 | Implausible goals are rejected | Goal more than 1000 m from the map origin, or not a number → rejected | Test |
| REQ-10 | Missing pose updates are reported | Warning in the terminal after more than 5 s without `pos_info` | Inspection, test on robot |
| REQ-11 | The robot stops when the laptop or network fails | Stop after 0.5 s without sensor data or 1 s without pose | Planned (watchdog on the Jetson) |

## Limits of these tests

- The tests check the logic, not the robot. Timing on the real network, DDS and SLAM behaviour are verified on the robot (see the deployment case).
- `ziel_pruefung.hpp` contains the checks from `slam_ziel.cpp` without the SDK. Next step: `slam_ziel.cpp` includes this header, so robot and tests use the same code.
