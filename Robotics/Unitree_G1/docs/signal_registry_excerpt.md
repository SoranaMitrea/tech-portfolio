# Signal Registry – Excerpt

The full registry documents the DDS interface of the Unitree G1 EDU (128 topics, plus services, message fields, hardware connections and all non-DDS paths) in the style of a DBC file. This excerpt shows the topics that were **verified on the running robot**.

Measurement conditions: 07.09.2026, robot hanging in its frame, Ethernet connection to the development PC.

## Status categories

| Status | Meaning |
|---|---|
| Verified | Measured on the device or publisher/subscriber checked on the running system |
| Verified inactive | Topic exists, but no one publishes (publisher count 0) |
| From topic list | Topic appears in the list, value and rate not checked |
| From header | Field exists in the SDK header file, value not checked |

## Verified topics

| Topic | Message type | Rate | Jitter p99 | Content |
|---|---|---|---|---|
| `/lowstate` | `unitree_hg/LowState` | 1041.4 Hz | 1.237 ms | Joints, body IMU, motor temperatures – main telemetry source |
| `/secondary_imu` | `unitree_hg/IMUState` | 1040.7 Hz | 1.250 ms | Second IMU in the pelvis (crotch) |
| `/lf/bmsstate` | `unitree_hg/BmsState` | 20.0 Hz | 51.2 ms | Battery data from the BMS |
| `/lf/mainboardstate` | `unitree_hg/MainBoardState` | 20.0 Hz | 51.7 ms | Fan states, temperatures, board values |
| `/lf/emergency_stop` | `unitree_go/Error` | event-based | – | Emergency stop state |
| `/lf/battery_alarm` | `std_msgs/String` | event-based | – | Battery warning |
| `/utlidar/cloud_livox_mid360` | `sensor_msgs/PointCloud2` | 9.97 Hz | – | LiDAR point cloud |
| `/utlidar/imu_livox_mid360` | `sensor_msgs/Imu` | 200.02 Hz | – | LiDAR IMU |
| `/audio_msg` | `std_msgs/String` | event-based | – | Recognized speech text, used for the voice dialogue |
| `/wirelesscontroller` | `unitree_go/WirelessController` | – | – | Remote control data, active |
| `/unitree/slam_relocation/odom` | `nav_msgs/Odometry` | – | – | Localization, active |
| `/global_map` | `nav_msgs/OccupancyGrid` | – | – | Occupancy grid, active |

## Verified inactive

| Topic | Finding |
|---|---|
| `/audiosender` | Publisher 0 – raw audio is not transmitted via DDS |
| `/lowstate_doubleimu` | Publisher 0 – message type unused |

## Findings from the registry

- The robot shares one software stack with other Unitree products: topics such as `/dog_odom` or `/frontvideostream` (Go2 type) appear in the list but are not relevant for the G1.
- The topics for the Dex3 hands exist, although no hands are fitted on this robot.
- The legs are commanded through the `/api/sport` service, not through joint angles.
