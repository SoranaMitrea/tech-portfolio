---

# Tech-portfolio

Welcome to my technical portfolio! Here I document my practical projects in *Arduino, Sensor Fusion, Artificial Intelligence (AI) and Robotics*.

---

**Latest:** Test automation & CI for the Unitree G1 → see Project no.6

---

*Project no.1* : **Automatic control of an air conditioning unit** *(temperature dependent)*

- Temperature and humidity measurement with a DHT11 sensor and display on an LCD 16x2
- Air conditioner control using an IR LED and a 2N2222 transistor

---

*Project no.2* : **Virtual Friction Sensor**

Real-time estimation of the tire-road friction coefficient using a combination of *empirical, classical ML and GenAI* approaches.

- Empirical model – calculates μ_truth from real driving data
- Classical AI model – uses *sensor fusion* (IMU, OBD2 data, temperature, humidity and pressure sensors) to predict friction in real time through supervised ML
- GenAI model – analyzes *acoustic patterns* from a microphone to detect surface conditions and refine the μ estimation

*The combined multi-layer architecture enables predictive and adaptive friction estimation for intelligent vehicle control and energy optimization.*

---

*Project no.3* : **Edge_ML_Inference**

A deterministic edge ML pipeline for real-time estimation of the tire-road friction coefficient from IMU, OBD2, GPS/RaceChrono and environmental data. Sensor values are synchronized, buffered, converted into features and processed directly on edge hardware with a lightweight ridge regression model. The focus is on a 100 ms cycle time, low latency, an interpretable model and later portability to embedded/automotive SoC platforms.

---

*Project no.4* : **Quantum Sensor Project**

Quantum sensor magnetic field simulation

- Simulation of a magnetic field sensor with noise, drift and spikes
- Rule-based anomaly detection (thresholds and heuristics)

---

*Project no.5* : **Robotics / Unitree G1**

Commissioning and system integration of a Unitree G1 EDU humanoid robot (23 DoF):

- DDS communication via `unitree_sdk2` (C++) and CycloneDDS, documented in a signal registry of 128 DDS topics
- Voice dialogue with a local LLM (Ollama), speech output and arm gestures
- Consent-based face recognition (OpenCV YuNet/SFace) with personalized greetings
- Direct LiDAR access (Livox MID-360) with mounting orientation corrected via IMU
- SLAM-based navigation to named goals with fixed safety checks before every goal

Details: [Robotics/Unitree_G1](Robotics/Unitree_G1)

---

*Project no.6* : **Test Automation & CI – Unitree G1**

Automated verification of the G1 goal navigation, derived from requirements:

- Requirement-based test cases with IDs (e.g. "go only with a pose ≤ 3 s old")
- pytest test suite for the pose bridge: data format, update rate, missing data
- Failure-case tests: connection loss, stale pose → motion command blocked
- GitHub Actions pipeline: tests run automatically on every commit
- Automatically generated test report with traceability to requirements

Details: [Robotics/Unitree_G1/testing](Robotics/Unitree_G1/testing)






