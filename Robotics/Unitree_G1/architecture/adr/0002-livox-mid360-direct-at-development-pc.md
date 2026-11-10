# ADR-0002: Livox MID-360 connected directly to the development PC

- **Status:** Draft
- **Date:** 2026-11-06
- **Scope:** LiDAR data path

## Context

The Livox MID-360 delivers a point cloud at 10 Hz and IMU data at 200 Hz. The sensor can reach the development PC through two paths: relayed via the Motion Controller, or directly over the internal network. The sensor is mounted upside down, which is not visible in the point cloud itself and was detected through the IMU data.

Network addresses (internal L2 network 192.168.123.0/24): development PC .51, Motion Controller .161, Jetson .164, LiDAR .120.

## Options considered

1. **Via the Motion Controller.** No extra connection at the development PC; point cloud rate measured at about 3 Hz.
2. **Direct to the development PC over UDP (chosen).** Point cloud rate measured at 10 Hz.

## Decision

The Livox MID-360 is connected directly to the development PC, using the Livox SDK2 and the Livox ROS 2 driver. The upside-down mounting is corrected with a roll correction in the driver configuration.

## Consequences

- Full point cloud rate (10 Hz) and IMU rate (200 Hz) are available for the room map and SLAM-related use.
- The LiDAR path depends on the development PC being present in the network; it does not run standalone on the robot.
- Mounting orientation is a configuration item, not a property of the data. A change in mounting requires re-checking the IMU gravity vector and the roll correction.
- Comparable measurements require the same operating mode and system state.

## Evidence

| Statement | Status |
|---|---|
| Point cloud 10 Hz direct, about 3 Hz via Motion Controller | measured or confirmed by source |
| IMU 200 Hz | measured or confirmed by source |
| Upside-down mounting detected via IMU, roll correction in driver | measured or confirmed by source |
| Cause of the reduced rate via the Motion Controller | assumed or unverified |
