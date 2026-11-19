# Architecture layer: Unitree G1 EDU

This folder describes the Unitree G1 EDU project at the level of an architecture: which functions and subsystems exist, which interfaces connect them, why the main decisions were taken, and which failures matter most. It sits on top of the working code and the measurements of the project. Public sources and own measurements only.

## Reading order

| Step | File | Question it answers |
|---|---|---|
| 1 | [`system_decomposition.md`](system_decomposition.md) | Which functions, subsystems and interfaces does the system have? What did the decomposition show? |
| 2 | [`communication_paths.md`](communication_paths.md) | How do control, voice, goals, LiDAR and pose travel between the development PC and the robot? |
| 3 | [`adr/`](adr/) | Why was each main decision taken, and which options were rejected? |
| 4 | [`fmea.md`](fmea.md) | Which failures matter most, and what is done about them? |
| 5 | [`safety_standards_overview.md`](safety_standards_overview.md) | Which safety standards frame a robot, a drone and a road vehicle, and why ISO 26262 is only a pattern here |

Five minutes: the figure and the findings in step 1, ADR-0001, and section 5 of the FMEA.

Images are in [`../docs/`](../docs/).

## Decisions

| ADR | Decision |
|---|---|
| [0001](adr/0001-control-and-voice-via-sdk2-cyclonedds.md) | Control, voice and navigation goals use unitree_sdk2 (C++) over CycloneDDS; ROS 2 only as read-only input |
| [0002](adr/0002-livox-mid360-direct-at-development-pc.md) | The Livox MID-360 stream goes directly to the development PC |
| [0003](adr/0003-language-model-produces-text-only.md) | The language model produces text only; motion is decided by fixed logic |
| [0004](adr/0004-release-rules-before-navigation-goals.md) | Release rules are checked before every navigation goal |

## How evidence is marked

- **C**: measured or confirmed by a source (own measurement, own code, manufacturer documentation).
- **A**: assumed or not verified.

The two are kept apart in every table. Where a statement rests on an assumption, it says so.

## Open points

- All documents are drafts. The ADRs are not yet accepted.
- Three measurements are outstanding and decide several FMEA rows: stopping delay and distance, a goal with the network cable unplugged (tethered), and navigation while the development PC LiDAR driver runs.
- The hazard ratings (S, F, P) and the FMEA ratings are initial estimates by one author and have not been reviewed by a second person.
- Nothing here is a safety case, and nothing claims conformity with any standard.
