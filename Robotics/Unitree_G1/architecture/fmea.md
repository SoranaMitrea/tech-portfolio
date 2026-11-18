# Operational FMEA: voice and navigation on the Unitree G1 EDU

- **Status:** Draft, ratings to be reviewed
- **Date:** 2026-11-18
- **Scope:** Goal-based navigation, voice dialogue and the release rules around them (ADR-0003, ADR-0004). Indoor use, one room, tethered tests.
- **Input:** [`system_decomposition.md`](system_decomposition.md) (functions F1 to F16, interfaces I1 to I13, findings 1 to 8)

## 1 Method and limits

This is an operational FMEA for prioritizing work on a test setup. It is **not** a hazard analysis and **not** a safety case. No ASIL, no safety goals, no claim of conformity with ISO 26262 or any other standard.

Scales, 1 to 5:

| Rating | 1 | 3 | 5 |
|---|---|---|---|
| **S** severity | no effect on motion | robot stops or leaves the path | fall or collision with a person or object |
| **O** occurrence | not seen, hard to imagine | seen once in the logs | seen repeatedly |
| **D** detection | detected before motion, automatic | detected during motion by the operator | not detectable in time |

RPN = S × O × D. The ratings are initial engineering estimates (**assumed or unverified**). They set the order of work, not a verdict. The column "Cause and evidence" separates observed facts (**C** = measured or confirmed by source) from assumptions (**A**).

## 2 FMEA table, sorted by RPN

| ID | Failure mode (function) | Effect | Cause and evidence | Existing control | S/O/D | RPN | Action |
|---|---|---|---|---|---|---|---|
| R1 | Obstacle in the path is not avoided (F14) | Collision with furniture, object or person; possible fall | The robot drives in a straight line. A balloon was not avoided (2026-10-07). Obstacles must be at least 50 cm high (manufacturer). **C**. Whether `slam_operate` avoids new obstacles was not tested. **A** | Dry run by default, confirmation, remote control in hand, tethered tests | 5/4/5 | 100 | Build stage 0 and 1 of the obstacle concept (watchdog, LiDAR stop zone). Until then: tethered tests, clear room, test whether `slam_operate` stops for obstacles |
| R2 | Pose is stale during a running goal (F8, I5) | Robot steers on a wrong position | Pose gaps up to 8 s while walking (2026-10-15). Release rule checks pose age only at the start. **C** | 3 s pose age rule at start | 4/4/4 | 64 | Watchdog that stops on pose age during the run (concept stage 0). Measure the real pose rate and gaps |
| R3 | Network drops while a goal runs (I1, I6) | Robot continues to the goal without supervision | Carrier off events logged (2026-10-15). **C**. Behaviour of a running goal without the sender is not tested. **A** | Remote control stop, tethered tests | 5/3/4 | 60 | Watchdog on the Jetson that sends a stop when pose, LiDAR or network are missing. Test: unplug the cable during a goal, with tether |
| R4 | Localization is lost, robot keeps its goal (F8) | Robot drives with a wrong map position | Localization was lost during a turn (2026-10-15). **C**. The read-only status command 1936 exists but is not part of the release rules. **C** | Pose age rule (does not detect wrong poses) | 5/3/4 | 60 | Check the relocalization status (1936) before every goal and during the run |
| R5 | LiDAR stream conflict between development PC and robot (F7, F8, I7) | Built-in SLAM loses its LiDAR data, localization fails | Stream goes to one receiver only; it stayed at the laptop until a robot restart (2026-10-07). **C**. Effect on the built-in SLAM while the PC receives: **A** | Check the pose before a goal | 4/3/4 | 48 | Decide one receiver per operating mode and write it down. Test: pose and navigation with the development PC driver running |
| R6 | `slam_operate` stops answering (F8, F10) | No goals, and no stop through the same service (1203) | Service did not answer after saving a map, after aborting a relocalization and after an app mode change; restart fixed it (2026-10). **C**. Cause open | Reply timeout of 8 s in the sender | 5/3/3 | 45 | Second stop path through `/api/sport` (velocity zero, `StopMove`) in the sender. Find the cause of the dropouts |
| R7 | Thresholds of the release rules are wrong (F11) | Rules too loose (unsafe goal accepted) or too tight (goal blocked) | 3 s and 3 m were set by judgement, not derived. **A** (ADR-0004) | Rules active, test limit 3 m | 3/3/5 | 45 | Derive the thresholds from the measured pose rate and the stopping behaviour; document in ADR-0004 |
| R8 | Operating state 501 or 801 behaves differently from 500 (F1, F11) | Unexpected speed or gait in an accepted state | Header file lists 500, 501 and 801 as states that accept motion commands. Meaning of 501 and 801 is not documented. **C** for the list, **A** for the behaviour | Rule accepts all three | 4/2/4 | 32 | Test 801 with the remote control; restrict the rule to 500 until proven |
| R9 | Two senders of motion at once (F10) | Conflicting commands from the app and the development PC | Manufacturer: do not use app navigation and the interface together. **C**. No technical lock. **A** | Operating rule | 4/2/4 | 32 | Add a check in the sender (is another sender active) or a visible interlock in the page |
| R10 | Model output reaches the motion path (F6, F9) | Unintended motion from free text | Design: model produces text, motion decided by fixed logic (ADR-0003). No code-level proof. **A** | Fixed rules, confirmation click | 4/2/4 | 32 | Code review or test showing that only the rule logic can trigger gestures and goals |
| R11 | Remote control not within reach or radio link lost (F15) | The last stop is missing | Remote control is the last stop. **C** (manufacturer). Link behaviour not tested. **A** | The goal sender asks for confirmation that the remote control is in hand | 5/2/3 | 30 | Add the remote control check to the commissioning checklist |
| R12 | Release rules skipped by test options (F11) | Goal sent with a stale pose or in a wrong operating state | `--ohne-pose` and `--ohne-zustand` skip two checks in the command-line tool. **C** | Options named in the help text, distance limit stays active | 4/2/3 | 24 | Remove the options from the default build or require a separate confirmation |
| R13 | Thermal state not observable (F16) | Overheating is not noticed | Fan state field stays 0 while the fan runs. **C**. Motor temperature is visible in `/lowstate`. **C** | Motor temperature telemetry | 3/2/3 | 18 | Add a temperature limit to the watchdog |
| R14 | LiDAR roll correction is wrong after remounting (F7) | Wrong map and pose | Mounted upside down, corrected in the driver (ADR-0002). **C** | IMU gravity check | 3/2/3 | 18 | Add the gravity check to the commissioning checklist |
| R15 | Wrong or missing voice recognition (F4) | Wrong target proposed, or no reaction | On-board recognition returned the wake word in Japanese script and "coach" for "couch" (2026-10-14). **C** | Page shows the target; goal only after "Losgehen" and confirmation | 2/4/2 | 16 | Whisper on the development PC (planned) |
| R16 | IP address of the USB Ethernet interface is lost (I1) | No state, no commands | The address is lost regularly (project note). **C** | Sender rejects the goal when the service is not reachable; network check before each session | 2/4/1 | 8 | Persistent address configuration |

## 3 What the table says

1. The top four items (R1 to R4) share one cause: nothing supervises a goal after it started. The release rules act only before the start.
2. One watchdog on the robot (pose, LiDAR, network, temperature, stop through `/api/sport`) lowers R1 to R4, R6 and R13 together. It is the largest single measure.
3. R5 and R6 depend on facts that are not yet measured. Two tests settle them: navigation with the development PC LiDAR driver running, and a goal with the network cable unplugged (with tether).

## 4 Not covered

- Hazards to people from the robot itself (falls, pinch points, stored energy): needs its own hazard analysis.
- Cybersecurity of the local server, tokens and network.
- Hardware failures inside the robot.
- Ratings were not reviewed by a second person.

## Sources

- [`system_decomposition.md`](system_decomposition.md) and the ADRs in [`adr/`](adr/)
- Own signal registry, own code, own notes on LiDAR and SLAM (2026-10-07), obstacle concept (2026-10-15), voice-to-goal (2026-10-14)
- Unitree documentation, SLAM and navigation services interface: https://support.unitree.com/home/en/G1_developer/slam_navigation_services_interface
