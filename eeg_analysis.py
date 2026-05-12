# Backend/analysis/eeg_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from eeg_constants import SFREQ, WINDOW_SIZE
from eeg_features import extract_psd_features

def analyze_eeg(eeg_signal):
    """
    Perform EEG analysis and return PSD features.
    """
    psd_features = extract_psd_features(eeg_signal, SFREQ, WINDOW_SIZE)
    return psd_features

def plot_eeg(eeg_signal, sfreq=SFREQ):
    """
    Plot raw EEG signal waveform.
    """
    times = np.arange(len(eeg_signal)) / sfreq
    plt.figure(figsize=(12, 4))
    plt.plot(times, eeg_signal)
    plt.title("EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Simulated EEG signal (1 channel, 10 seconds)
    eeg_signal = np.random.randn(SFREQ * 10)
    features = analyze_eeg(eeg_signal)
    print("PSD Features:", features)
    plot_eeg(eeg_signal)