# Test Automation & CI – Unitree G1 *(in progress)*

Automated verification of the G1 goal navigation
([deployment case](../README.md)), derived from requirements.

## Approach

- Each test case references a requirement ID (REQ-xx)
- Logic is tested without the robot (pure Python, no ROS needed)
- Failure cases are simulated: stale pose, missing data, connection loss
- Tests run automatically on every commit (GitHub Actions)

## Planned structure

    testing/
    ├── requirements.md      requirements with IDs and acceptance criteria
    ├── src/                 testable logic extracted from the pose bridge
    ├── tests/               pytest test cases
    └── report/              generated test report with traceability
