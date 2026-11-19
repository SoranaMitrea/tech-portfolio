# Communication paths: unitree_sdk2, CycloneDDS and ROS 2

Control, voice and navigation goals run through unitree_sdk2 (C++) and CycloneDDS without ROS 2. ROS 2 is used only as read-only input: the LiDAR point cloud (Livox MID-360 via Livox SDK2 and the Livox ROS 2 driver) and the SLAM pose (see ADR-0001, ADR-0002).

```mermaid
flowchart LR
    subgraph PC["Development PC"]
        APP["C++ program<br/>unitree_sdk2: control, voice, goals"]
        DDS["CycloneDDS<br/>native DDS, no ROS 2"]
        LSDK["Livox SDK2<br/>vendor SDK, UDP"]
        LDRV["Livox ROS 2 driver<br/>ROS 2 (rclpy), read-only"]
        POSE["ROS 2 pose reader<br/>rclpy, read-only"]
        APP --> DDS
        LSDK --> LDRV
    end
    subgraph ROBOT["Robot"]
        MC["Motion Controller<br/>joints, mic, speaker"]
        SLAM["Unitree SLAM service<br/>slam_operate, localization"]
        LI["Livox MID-360<br/>10 Hz, IMU 200 Hz"]
    end
    DDS <-- "DDS" --> MC
    DDS -- "goals (api_id 1102)" --> SLAM
    SLAM -- "/slam_info" --> POSE
    LI -- "UDP" --> LSDK

    classDef native fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef rosnode fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    class APP,DDS,MC,SLAM native
    class LSDK,LDRV,POSE,LI rosnode
```

| Colour | Path |
|---|---|
| Green | Native DDS path (control, voice, navigation goals), no ROS 2 |
| Purple | LiDAR and SLAM-pose input, read-only. The Livox SDK2 is the LiDAR vendor's SDK, not unitree_sdk2. |

Status:

- The SLAM service `slam_operate` runs on the Motion Controller (192.168.123.161), observed 2026-10-09 (measured or confirmed by source).
- The pose (`rt/slam_info`) is also a DDS topic. Reading it through unitree_sdk2 instead of ROS 2 is expected to work but was not tried (assumed or unverified).
