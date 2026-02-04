import numpy as np
import pandas as pd

fs = 1000                 # Abtastrate (Hz)
T = 10                    # Sekunden
t = np.linspace(0, T, fs*T)

integration_time = 1e-2   # 1 ms Integrationszeit typisch für Quantensensor 
                          # das verstärkt das Rauschen um Wurzel aus 1000 ~ 31x - physikalisch korrekt (für Demo kann man 1e-2 nehmen)
                    
# echtes Signal (z.B. Motorstrom / Feldänderung)

def generate_signal(noise_scale=1.0, spike=False):

    B_signal = 50e-6 * np.sin(2*np.pi*2*t) # das ideal Signal- hier Sinus mit 2 Hz und Amplitude 50muT

    base_noise = 3e-6 * noise_scale     # für Demo Zwecke 5e-7, realistisch aber 3e-6
    noise = np.random.normal(0, base_noise/np.sqrt(integration_time), len(t))
                                        # zufälliges Messrauschen (Gaussian/ Normalverteilung) wie ein echter Quantensensor elektronik + Photonrauschen, Quantum Shot Noise
                                        # das Ergebnis ist ein Vektor noise[t]
                                        # np.random.normal(mean, std, size)
                                        #              0.3 --> 0.9 muT --> std ~ 28.4 muT (Noise für ein "good" Signal) 
                                        # Noise Skala  1.0 --> 1.0 muT --> std ~ 94.9.4 muT (Noise für ein "fault" Signal)
                                        #              2.0 --> 6.0 muT --> std ~ 189.7 muT (Noise für ein "noisy" Signal)
                                        # sqrt(integration_time) aus Quantensensor Physik (je länger man integriert, dest weniger Rauschen)
                                        # Noise 1/sqrt(T) --> 1/sqrt (1e-3 ~ 31.6)
                                        # Für jedes Sample len(t) wird ein noise ereugt: noise ~ N(0, Base_noise/sqrt(T))
    drift = 1e-6 * t                  # linearer Offset über Zeit (reale Ursachen: Temp. Änderung, Alterung, Versorgungsspannung driftet)

    spike_arr = np.zeros_like(t)
    if spike:
        spike_arr[3000] = 18e-5   # es wird ein Spike  muT addiert; ein 200ms Magnetfeldsprung non +20T;   
        spike_arr[3001] = 20e-6
        spike_arr[3002] = -10e-6
        spike_arr[3003] =15e-6
        
    B = B_signal + noise + drift + spike_arr   # realistisches Magnetfeldsignal

    return pd.DataFrame({"time": t, "B_field_T": B})

from pathlib import Path
BASE = Path(__file__).resolve().parents[1]   # Quantensensor/
OUTDIR = BASE / "1_data" / "raw"
OUTDIR.mkdir(parents=True, exist_ok=True)

good = generate_signal(0.3, False)   # signal_good --> Noise wenig, Spike nein
noisy = generate_signal(2.0, False)  # signal_noisy --> Noise viel, Spike nein
fault = generate_signal(1.0, True)   # signal_fault --> Noise mittel, Spike ja

good.to_csv(OUTDIR / "signal_good.csv", index=False)   
noisy.to_csv(OUTDIR / "signal_noisy.csv", index=False) 
fault.to_csv(OUTDIR / "signal_fault.csv", index=False) 

print("3 CSV files generated.")

import matplotlib.pyplot as plt

# -------------------------------------------------
# 5 Plots erzeugen (ohne generate_signal zu ändern)
# -------------------------------------------------

# 1) B_signal (ideales Signal)
B_signal = 50e-6 * np.sin(2*np.pi*2*t)

plt.figure()
plt.plot(t, B_signal)
plt.title("1) B_signal (ideal)")
plt.xlabel("time [s]")
plt.ylabel("B [T]")
plt.grid(True)

# 2) B (ein Beispiel-Messsignal, neu erzeugt – ohne Spike)
_, B_example = None, None
tmp = generate_signal(1.0, False)
B_example = tmp["B_field_T"]

plt.figure()
plt.plot(t, B_example)
plt.title("2) B (Messsignal, noise_scale=1.0)")
plt.xlabel("time [s]")
plt.ylabel("B [T]")
plt.grid(True)

# 3) signal_good.csv
plt.figure()
plt.plot(good["time"], good["B_field_T"])
plt.title("3) signal_good")
plt.xlabel("time [s]")
plt.ylabel("B [T]")
plt.grid(True)

# 4) signal_fault.csv
plt.figure()
plt.plot(fault["time"], fault["B_field_T"])
plt.title("4) signal_fault (mit Spike)")
plt.xlabel("time [s]")
plt.ylabel("B [T]")
plt.grid(True)

# 5) signal_noisy.csv
plt.figure()
plt.plot(noisy["time"], noisy["B_field_T"])
plt.title("5) signal_noisy")
plt.xlabel("time [s]")
plt.ylabel("B [T]")
plt.grid(True)

plt.show()
