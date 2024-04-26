import librosa
import numpy as np

# Load speech recording with noise
audio_file = 'speech_with_noise.wav'
y, sr = librosa.load(audio_file, sr=None)

# Apply spectral subtraction for noise reduction
stft = librosa.stft(y)
stft_denoised = np.maximum(0, np.abs(stft) - 2) * np.exp(1j * np.angle(stft))
y_denoised = librosa.istft(stft_denoised)

# Save denoised speech recording
denoised_audio_file = 'speech_denoised.wav'
librosa.output.write_wav(denoised_audio_file, y_denoised, sr)
