import os

import librosa
from prepare_data import create_data
from data_display import make_plot_spectrogram, make_3plots_spec_voice_noise, make_3plots_phase_voice_noise, make_3plots_timeseries_voice_noise
import numpy as np 
import matplotlib.pyplot as plt 
from unet_2 import UNET
import torch
from data_tools import magnitude_db_and_phase_to_audio, scale_in, scale_ou, inv_scaled_in, inv_scaled_ou, audio_files_to_numpy, numpy_audio_to_matrix_spectrogram
from speech_data import Dataset, SpeechDataset
from torchvision import transforms
import soundfile as sf

noise_dir = 'dataset/Test/noise'
voice_dir = 'dataset/Test/clean_voice'
path_save_time_serie = 'dataset/Test/time_serie/'
path_save_sound = 'dataset/Test/sound/'
path_save_spectrogram = 'dataset/Test/spectrogram/'
sample_rate = 8000
min_duration = 1

play_time = 1 #seconds
frame_length = 8064*play_time #8064
hop_length_frame = 8064*play_time #8064
hop_length_frame_noise = 5000*play_time
nb_samples = 10
n_fft = 255
hop_length_fft = 63*play_time

dim_square_spec = int(n_fft/2)+1

model = UNET()
model.load_state_dict(torch.load('model/new_model_6.pth', map_location=torch.device('cpu')))
transform = transforms.Compose([transforms.ToTensor()])

sound = os.listdir(path_save_sound)

np_sound = audio_files_to_numpy(path_save_sound, sound, sample_rate, frame_length, hop_length_frame, min_duration)

n_sound = np_sound.shape[0]

result_sound = []
noise = [] #use to plot nosie
v_noise = [] #use to plt noisy voice
voice = [] #use to plot clean voice

for i in range(n_sound):
    m_amp_db_sound, m_phase_db_sound = numpy_audio_to_matrix_spectrogram(
        np_sound[i:i+1], 
        dim_square_spec,
        n_fft,
        hop_length_fft
    )
    
    voice_noise = scale_in(m_amp_db_sound)
    img = torch.tensor(voice_noise)
    
    pred = model(img[None, ...].float())
    pred = inv_scaled_ou(pred)

    voice_noise = inv_scaled_in(voice_noise)
    clean_pred = voice_noise - pred.detach().numpy()[0]
    
    scale_volume = 15 #volume scale
    voice.append(clean_pred * scale_volume)
    v_noise.append(voice_noise)
    noise.append(pred.detach().numpy()[0])

    audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, clean_pred, m_phase_db_sound)

    result_sound.extend(audio_reconstruct[0]*scale_volume)

result_sound = np.array(result_sound)
sf.write('dataset/Test/result_sound/' + 'result.wav', result_sound, sample_rate)

test_id = 0

make_3plots_spec_voice_noise(v_noise[test_id][0], noise[test_id][0], voice[test_id][0], sample_rate, hop_length_fft)
plt.show()