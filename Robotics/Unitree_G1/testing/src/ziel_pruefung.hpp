// ziel_pruefung.hpp - safety checks for a navigation goal (command 1102), without the Unitree SDK.
//
// The checks are taken unchanged from slam_ziel.cpp (pose_lesen, zahl_nach and the check block
// before sending). Kept free of SDK and DDS so they can be tested in CI without the robot.
//
// Order of the checks, as in slam_ziel.cpp:
//   1. limit: max_m is capped at 10 m (manufacturer limit), max_m <= 0 is an error
//   2. pose:  pose must be valid and at most 3 s old, otherwise rejected (unless ohne_pose)
//   3. distance from pose to goal must not exceed max_m
//   4. plausibility: goal further than 1000 m from the map origin is rejected
//   5. state: fsm_id must be 500, 501 or 801 (walking), otherwise rejected (unless ohne_zustand)

#pragma once
#include <cmath>
#include <cstdlib>
#include <string>

namespace g1 {

static const double ABSOLUTE_GRENZE_M = 10.0;
static const double POSE_ALTER_MAX_S = 3.0;

struct Pose { bool gueltig = false; double x = 0, y = 0, yaw_grad = 0, alter_s = 1e9; };

struct Ergebnis { bool freigabe = false; std::string grund; };

inline bool zahl_nach(const std::string& s, const std::string& schluessel, double& wert)
{
    size_t p = s.find("\"" + schluessel + "\"");
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && s[p] == ' ') ++p;
    char* ende = nullptr;
    wert = std::strtod(s.c_str() + p, &ende);
    return ende != s.c_str() + p;
}

// Same as pose_lesen() in slam_ziel.cpp, but reads from text and takes the current time as a parameter.
inline Pose pose_aus_text(const std::string& s, double jetzt)
{
    Pose p;
    double zeit = 0;
    if (!zahl_nach(s, "x", p.x) || !zahl_nach(s, "y", p.y) || !zahl_nach(s, "yaw_grad", p.yaw_grad) || !zahl_nach(s, "zeit", zeit))
        return p;
    p.alter_s = jetzt - zeit;
    p.gueltig = true;
    return p;
}

inline Ergebnis ziel_pruefen(double zx, double zy, const Pose& p, bool ohne_pose,
                             int fsm, bool ohne_zustand, double max_m)
{
    if (max_m > ABSOLUTE_GRENZE_M) max_m = ABSOLUTE_GRENZE_M;
    if (max_m <= 0) return {false, "max_not_positive"};
    if (p.gueltig && p.alter_s <= POSE_ALTER_MAX_S) {
        const double d = std::hypot(zx - p.x, zy - p.y);
        if (d > max_m) return {false, "too_far"};
    } else if (!ohne_pose) {
        return {false, "no_fresh_pose"};
    }
    if (std::hypot(zx, zy) > 1000.0) return {false, "implausible"};
    if (!ohne_zustand && !(fsm == 500 || fsm == 501 || fsm == 801)) return {false, "not_walking"};
    return {true, "ok"};
}

}  // namespace g1
