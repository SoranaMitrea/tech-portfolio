# Test Automation & CI – Unitree G1

Automated verification of the G1 goal navigation (see the [G1 project](../README.md)), derived from requirements.

[![G1 tests](https://github.com/SoranaMitrea/tech-portfolio/actions/workflows/g1-tests.yml/badge.svg)](https://github.com/SoranaMitrea/tech-portfolio/actions/workflows/g1-tests.yml)

## What is tested

| Part | Code | Language |
|---|---|---|
| Pose bridge: reads `/slam_info`, writes the robot pose to a file | `src/g1_slam_pose.py` | Python |
| Safety checks before a navigation goal is sent (from `slam_ziel.cpp`) | `src/ziel_pruefung.hpp` | C++ |

## Approach

- Requirements with IDs and acceptance criteria: [requirements.md](requirements.md)
- Every test is linked to a requirement ID (`@pytest.mark.req("REQ-06")`)
- Logic is tested without the robot: Python without ROS, C++ without the Unitree SDK
- Failure cases are simulated: stale pose, pose from the future, malformed messages, second bridge, robot not walking, NaN goal
- GitHub Actions runs all tests on every commit and builds the [traceability report](report/test_report.md)

## Results

47 tests passed. Two tests document open findings (expected failures):

- **F-01** (REQ-06): a pose with a timestamp in the future has a negative age and counts as fresh.
- **F-02** (REQ-09): the goal `nan nan` passes all checks, because every comparison with NaN is false.

Both were found by these tests. The fix is a check for a negative age and for `std::isfinite`.
After the fix, the two tests turn into normal passing tests.

## Run locally

    cd Robotics/Unitree_G1/testing
    pip install pytest
    python -m pytest tests -v --junitxml=report/junit.xml
    python tools/report.py report/junit.xml requirements.md report/test_report.md

Requires Python 3 and g++ (Linux).

## Limits

The tests check the logic, not the robot. Network timing, DDS and SLAM behaviour are verified on the robot.
`ziel_pruefung.hpp` holds the checks from `slam_ziel.cpp` without the SDK; next step is that `slam_ziel.cpp`
includes this header, so the robot and the tests use the same code.

## Structure

    testing/
    ├── requirements.md        requirements with IDs and acceptance criteria
    ├── src/                   code under test
    ├── tests/                 pytest test cases (Python and C++)
    ├── tools/report.py        builds the traceability report
    └── report/                test_report.md
    .github/workflows/g1-tests.yml   (in the repository root) CI pipeline
