# ADR-0004: Release rules before every navigation goal

- **Status:** Draft
- **Date:** 2026-11-06
- **Scope:** Navigation goals sent to the G1

## Context

Navigation goals in the LiDAR map are sent from a local browser page. Localization comes from the Unitree SLAM service; the current pose is read via ROS 2 (`/slam_info`). Goals themselves are sent with unitree_sdk2 to the `slam_operate` service (api_id 1102) over DDS. A goal sent with a stale pose, in an unsuitable operating state, or at an excessive distance can lead to unintended robot motion.

## Options considered

1. **No pre-check.** Goals are sent as entered.
2. **Operator judgment only.** The operator is responsible for checking state and pose.
3. **Fixed release rules in the sender (chosen).** Every goal passes the same checks, independent of who or what proposed it.

## Decision

Before every goal, the sender checks:

| Rule | Value |
|---|---|
| Operating state (fsm_id) | 500, 501 or 801 |
| Pose age | at most 3 s |
| Distance to goal | at most 3 m (test limit); absolute limit 10 m |
| Execution mode | dry run is the default; a goal is sent only after explicit confirmation |

The browser page runs locally only and asks for a new password at every start.

## Consequences

- A goal is rejected if any rule fails; there is no override path through the language model (see ADR-0003).
- The 3 s pose age limit couples the safety rule to the SLAM pose rate and to the ROS 2 path. A degraded pose stream blocks navigation by design.
- The 3 m test limit and the 10 m absolute limit are separate parameters; moving from test to normal operation changes only the first.
- The command-line tool offers two test options (`--ohne-pose`, `--ohne-zustand`) that skip the pose check and the state check. The rules are therefore not enforced on every use path; the distance limit stays active.
- The rules reduce risk but do not replace a hazard analysis. The operational FMEA (falls, collision, communication loss, wrong goal) is a separate work item.

## Evidence

| Statement | Status |
|---|---|
| Rule set implemented in the sender as described | measured or confirmed by source |
| Motion commands are accepted only at fsm_id 500, 501 and 801 | measured or confirmed by source (Unitree header file, own state overview) |
| Meaning of fsm_id 501 and 801 | assumed or unverified: not described in the header file; 801 needs a check with the remote control |
| Absolute distance limit of 10 m | measured or confirmed by source (manufacturer limit, as noted in the code) |
| Command-line tool can skip the pose and state checks | measured or confirmed by source (own code) |
| Test limits 3 s pose age and 3 m distance derived from measured pose rate (about 1.2 Hz) and stopping behavior | assumed or unverified: pose rate noted in code, no derivation documented |
