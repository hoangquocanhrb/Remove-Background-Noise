import os

import librosa
from prepare_data import create_data
from data_display import make_plot_spectrogram, make_3plots_spec_voice_noise, make_3plots_phase_voice_noise, make_3plots_timeseries_voice_noise
import numpy as np 
import matplotlib.pyplot as plt 

noise_dir = 'engDataset/Train/noise/'
voice_dir = 'engDataset/Train/clean_voice/'
path_save_time_serie = 'engDataset/Train/time_serie/'
path_save_sound = 'engDataset/Train/sound/'
path_save_spectrogram = 'engDataset/Train/spectrogram/'
sample_rate = 8000
min_duration = 1
frame_length = 8064
hop_length_frame = 8064
hop_length_frame_noise = 5000
nb_samples = 12000
n_fft = 255
hop_length_fft = 63

create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
        min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft)
