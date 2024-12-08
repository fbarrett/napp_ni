import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def read_audio(file_path):
    rate, data = wav.read(file_path)
    return rate, data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def simulate_cochlear_response(file_path, lowcut=300, highcut=3400, order=5):
    fs, data = read_audio(file_path)
    if data.ndim > 1:
        data = data[:, 0]  # Use only one channel if stereo
    filtered_data = bandpass_filter(data, lowcut, highcut, fs, order)
    time = np.arange(len(filtered_data)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, filtered_data)
    plt.title('Simulated Cochlear Nerve Response')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

# Example usage
simulate_cochlear_response('path_to_audio_file.wav')