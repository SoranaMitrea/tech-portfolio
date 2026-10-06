// read_state.cpp — reads LowState from the Unitree G1 and prints IMU and arm values.
// Read-only: no publisher, no motor commands. Safe to run with the robot hanging or standing.
//
// Location:  ~/unitree_sdk2/example/g1/low_level/read_state.cpp
//
// Add to ~/unitree_sdk2/example/g1/CMakeLists.txt:
//   add_executable(g1_read_state low_level/read_state.cpp)
//   target_link_libraries(g1_read_state unitree_sdk2)
//
// Build:
//   cd ~/unitree_sdk2/build && cmake .. && make g1_read_state
//
// Run:
//   export CYCLONEDDS_URI=file://$HOME/.config/cyclonedds/g1.xml
//   ~/unitree_sdk2/build/bin/g1_read_state

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <thread>
#include <iostream>
#include <iomanip>

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

int main(int argc, char** argv)
{
    // DDS domain 0; optional network interface name as first argument
    ChannelFactory::Instance()->Init(0, argc > 1 ? argv[1] : "");
    auto sub = std::make_shared<ChannelSubscriber<LowState_>>("rt/lowstate");
    LowState_ s;
    bool got = false;
    // Callback runs for every incoming message (about 1000 Hz) and copies the latest state
    sub->InitChannel([&](const void* msg) {
        s = *(const LowState_*)msg;
        got = true;
    });
    while (true) {
        if (got) {
            std::cout << std::fixed << std::setprecision(3);
            // Roll, pitch, yaw of the torso IMU in radians
            std::cout << "RPY: " << s.imu_state().rpy()[0] << " "
                      << s.imu_state().rpy()[1] << " "
                      << s.imu_state().rpy()[2] << "\n";
            // Right arm, motor indices 22–28 (joint angles in radians).
            // On the 23-DoF version, indices 27 and 28 are not fitted and read 0.000.
            std::cout << "Right arm: ";
            for (int i = 22; i < 29; i++)
                std::cout << s.motor_state()[i].q() << " ";
            std::cout << "\n---\n";
        }
        // Limit output to two lines per second
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
