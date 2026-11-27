# Test report – G1 goal navigation

Generated 2026-11-27 14:39 UTC from `report/junit.xml`.

| Status | Count |
|---|---|
| passed | 47 |
| failed | 0 |
| finding | 2 |
| skipped | 0 |

## Traceability

| ID | Requirement | Method | Tests | Passed | Findings | Result |
|---|---|---|---|---|---|---|
| REQ-01 | The bridge only takes pose messages | Test | 5 | 5 | 0 | passed |
| REQ-02 | Malformed messages must not stop the bridge | Test | 7 | 7 | 0 | passed |
| REQ-03 | The heading is given in degrees | Test | 4 | 4 | 0 | passed |
| REQ-04 | Other programs never read a half-written pose | Test | 2 | 2 | 0 | passed |
| REQ-05 | Only one pose bridge runs at a time | Test | 2 | 2 | 0 | passed |
| REQ-06 | A goal is only sent with a fresh pose | Test | 10 | 9 | 1 | passed, with open finding |
| REQ-07 | A goal is only sent within the distance limit | Test | 9 | 9 | 0 | passed |
| REQ-08 | A goal is only sent while the robot walks | Test | 8 | 8 | 0 | passed |
| REQ-09 | Implausible goals are rejected | Test | 2 | 1 | 1 | passed, with open finding |
| REQ-10 | Missing pose updates are reported | Inspection, test on robot | 0 | 0 | 0 | see method |
| REQ-11 | The robot stops when the laptop or network fails | Planned (watchdog on the Jetson) | 0 | 0 | 0 | see method |

## Findings (open, documented as expected failures)

- **REQ-06** `test_pose_from_future_rejected`: FINDING F-01: a pose with a timestamp in the future has a negative age and is accepted as fresh
- **REQ-09** `test_nan_goal_rejected`: FINDING F-02: goal "nan nan" passes all checks, because every comparison with NaN is false
