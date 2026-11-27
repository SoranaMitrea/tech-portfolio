// ziel_cli.cpp - small command-line wrapper around ziel_pruefung.hpp so pytest can call the C++ checks.
//
//   ziel_cli <zx> <zy> <pose_json> <jetzt> <fsm> <max_m> [--ohne-pose] [--ohne-zustand]
//
// Prints FREIGABE or ABGELEHNT <reason>. Exit code 0 = released, 5 = rejected (as slam_ziel.cpp).
// Build: g++ -std=c++17 -O2 -o ziel_cli ziel_cli.cpp

#include "ziel_pruefung.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s zx zy pose_json jetzt fsm max_m [--ohne-pose] [--ohne-zustand]\n", argv[0]);
        return 1;
    }
    bool ohne_pose = false, ohne_zustand = false;
    for (int i = 7; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ohne-pose")) ohne_pose = true;
        else if (!std::strcmp(argv[i], "--ohne-zustand")) ohne_zustand = true;
    }
    // std::atof as in slam_ziel.cpp, so inputs such as "nan" behave exactly as on the robot
    const double zx = std::atof(argv[1]), zy = std::atof(argv[2]);
    const g1::Pose p = g1::pose_aus_text(argv[3], std::atof(argv[4]));
    const g1::Ergebnis e = g1::ziel_pruefen(zx, zy, p, ohne_pose, std::atoi(argv[5]), ohne_zustand, std::atof(argv[6]));
    if (e.freigabe) { std::printf("FREIGABE\n"); return 0; }
    std::printf("ABGELEHNT %s\n", e.grund.c_str());
    return 5;
}
