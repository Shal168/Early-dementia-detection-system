
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import uuid

SFREQ = 256
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_eeg_from_bytes(content):
    from io import BytesIO
    try:
        eeg_data = np.load(BytesIO(content))
    except:
        from io import StringIO
        eeg_data = np.loadtxt(StringIO(content.decode()), delimiter=",")

    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    return eeg_data


def generate_eeg_image(eeg_data):
    seconds = 5
    num_channels = min(3, eeg_data.shape[1])

    eeg_data = eeg_data[:int(seconds * SFREQ), :num_channels]
    time = np.arange(eeg_data.shape[0]) / SFREQ

    signal = eeg_data[:, 0]
    nperseg = min(len(signal), SFREQ * 2)
    freqs, psd = welch(signal, fs=SFREQ, nperseg=nperseg)

    plt.figure(figsize=(10, 8))

    # EEG waveform
    plt.subplot(3, 1, 1)
    for i in range(num_channels):
        plt.plot(time, eeg_data[:, i] + i * 100)
    plt.title("EEG Waveform")
    plt.grid(True)

    # Single channel
    plt.subplot(3, 1, 2)
    plt.plot(time, signal)
    plt.title("Single Channel")
    plt.grid(True)

    # PSD
    plt.subplot(3, 1, 3)
    plt.plot(freqs, psd)
    plt.title("Power Spectral Density")
    plt.xlim(0, 50)
    plt.grid(True)

    plt.tight_layout()

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    plt.close()

    return filename