import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def simulate_auditory_nerve_response(sound_file):
    # Read the sound file
    sample_rate, signal = wav.read(sound_file)
    
    # Normalize the signal
    signal = signal / np.max(np.abs(signal))
    
    # Define parameters for the auditory nerve model
    # These parameters can be adjusted based on the specific model being used
    filter_bank = np.array([100, 200, 400, 800, 1600, 3200, 6400])  # Example filter bank frequencies in Hz
    response = np.zeros((len(filter_bank), len(signal)))
    
    # Simulate the response of each filter in the filter bank
    for i, freq in enumerate(filter_bank):
        # Create a bandpass filter centered at the current frequency
        b, a = scipy.signal.butter(2, [freq - 50, freq + 50], btype='band', fs=sample_rate)
        filtered_signal = scipy.signal.lfilter(b, a, signal)
        
        # Rectify and low-pass filter the signal to simulate the auditory nerve response
        rectified_signal = np.abs(filtered_signal)
        b, a = scipy.signal.butter(2, 50, btype='low', fs=sample_rate)
        response[i, :] = scipy.signal.lfilter(b, a, rectified_signal)
    
    # Plot the simulated auditory nerve response
    plt.figure(figsize=(10, 6))
    for i, freq in enumerate(filter_bank):
        plt.plot(response[i, :], label=f'{freq} Hz')
    plt.xlabel('Time (samples)')
    plt.ylabel('Response')
    plt.title('Simulated Auditory Nerve Response')
    plt.legend()
    plt.show()

# Example usage
simulate_auditory_nerve_response('path_to_sound_file.wav')