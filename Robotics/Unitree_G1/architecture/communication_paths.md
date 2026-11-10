# Communication paths: unitree_sdk2, CycloneDDS and ROS 2

Control and voice run through unitree_sdk2 (C++) and CycloneDDS without ROS 2. ROS 2 is used only as read-only input for the LiDAR point cloud and the SLAM pose (see ADR-0001, ADR-0002).

```mermaid
flowchart LR
    subgraph PC["Development PC"]
        APP["C++ program<br/>unitree_sdk2: control, voice"]
        DDS["CycloneDDS<br/>native DDS, no ROS 2"]
        ROS["ROS 2 (rclpy)<br/>Livox driver, pose reader"]
        APP --> DDS
    end
    subgraph ROBOT["Robot"]
        MC["Motion Controller<br/>joints, mic, speaker"]
        SLAM["Unitree SLAM service<br/>localization"]
        LI["Livox MID-360<br/>10 Hz, IMU 200 Hz"]
    end
    DDS <-- "DDS" --> MC
    SLAM -- "/slam_info" --> ROS
    LI -- "UDP" --> ROS

    classDef native fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef rosnode fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    class APP,DDS,MC native
    class ROS,SLAM,LI rosnode
```

| Colour | Path |
|---|---|
| Green | Native DDS path (control, voice), no ROS 2 |
| Purple | ROS 2 path, read-only input (point cloud, pose) |

Status: where the SLAM service runs on the robot is not documented here (assumed or unverified).
