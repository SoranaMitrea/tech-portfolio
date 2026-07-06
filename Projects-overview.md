📄 **Hinweis:** Manche PDFs zeigen in GitHubs Vorschau (besonders in Firefox) den Fehler "Error rendering embedded code". Einfach oben rechts bei der Datei auf **"⋯" → Download** klicken, um sie zu öffnen.

 -------------------------------------------------------------------------------------------------------------------------------------------------------------
 -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tech-portfolio

Welcome to my technical portfolio! Here I document my practical projects in *Arduino, Sensor Fusion, and Artificial Intelligence (AI)*, etc.  

 -------------------------------------------------------------------------------------------------------------------------------------------------------------


*Project no.1* : **Automatic control of an air conditioning unit** *(temperature dependent)*
- Temperature and humidity measurement with DHT11 Temperature and Humidity sensor and display on LCD 16x2  
- Air conditioner control using IR LED + 2N2222 transistor
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------

*Project no.2* : **Virtual Friction sensor**

Real time estimation of tire-road friction coefficient using a combination of *empirical, classical ML and GenAI* approach
-   Empirical Model - calculates µtruth from real driving data
-   Classical AI Model - uses *sensor fusion* (IMU sensor, OBD2 data, Temp.-Humid.-Press. sensor) to predict friction in real time through supervised ML
-   GenAI Model - analyzes *acoustic patterns* from a microphone to detect surface conditions and refine µ estimation.

  *The combined multi-layer architectures enables predictive and adaptive friction estimation for intelligent vehicle control and energy optimization.*

---------------------------------------------------------------------------------------------------------------------------------------------------------------

*Project no.3* : **Edge_ML_Inference**

Project Edge ML Inference entwickelt eine deterministische Edge-ML-Pipeline zur Echtzeit-Schätzung des Reifen-Straßen-Reibwerts aus IMU-, OBD2-, GPS/RaceChrono- und Umweltdaten.
Die Sensorwerte werden synchronisiert, gepuffert, in Features umgewandelt und mit einem leichtgewichtigen Ridge-Regression-Modell direkt auf Edge-Hardware verarbeitet.
Der Fokus liegt auf 100-ms-Zykluszeit, niedriger Latenz, interpretierbarem Modell und späterer Übertragbarkeit auf Embedded-/Automotive-SoC-Plattformen.


---------------------------------------------------------------------------------------------------------------------------------------------------------------

*Project no.4* : **Quantensensor project**

Quantensensor  Magnetfelsimulation
-  Simulation eines Magnetfeldsensors mit Noise, Drift und Spikes
-  regelbasierte Anomalieerkennung (Threshold und Heuristiken)

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Project no.5 : **Robotics/ AI_Platform**

A modular ROS2-based robotics platform for autonomous service and event robots, combining sensor fusion, perception, navigation, human-robot interaction and AI modules such as object detection, speech interaction and face recognition.

*The project demonstrates a system-level architecture for intelligent robots using Edge/Cloud integration, simulation, digital twins and AI-based decision support.*

