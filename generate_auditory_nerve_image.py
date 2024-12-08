import librosa
import numpy as np

import librosa.display
import matplotlib.pyplot as plt

def generate_auditory_nerve_image(audio_path, output_image_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.close()

# Example usage
generate_auditory_nerve_image('/Users/fbarrett/Documents/_studies/2300_PRoMISS/Study1/_music/003/01 Beauty And A Beat.wav', 'output_image.png')