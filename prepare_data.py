import os
import librosa
from data_tools import audio_files_to_numpy
from data_tools import blend_noise_randomly, numpy_audio_to_matrix_spectrogram
import numpy as np
import soundfile as sf

def read_voice_files(path):
    folders = os.listdir(path)
    audio_files = []
    for sub1 in folders:
        m = os.listdir(path + '/' + str(sub1))
        for sub2 in m:
            audio = os.listdir(path + '/' + str(sub1) + '/' + str(sub2))
            for f in audio:
                if f.endswith(".flac"):
                    audio_files.append(str(sub1) + '/' + str(sub2) + '/' + str(f))
    return audio_files

def create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    
    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)
    
    nb_voice_files = len(list_voice_files)
    nb_noise_files = len(list_noise_files)

    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate, 
                                frame_length, hop_length_frame_noise, min_duration)
    voice = audio_files_to_numpy(voice_dir, list_voice_files, sample_rate,
                                frame_length, hop_length_frame, min_duration)

    prob_voice, prob_noise, prob_noisy_voice = blend_noise_randomly(
        voice, noise, nb_samples, frame_length)
    
    noisy_voice_long = prob_noisy_voice.reshape(1, nb_samples * frame_length)
    sf.write(path_save_sound + 'noisy_voice_long.wav', noisy_voice_long[0, :], sample_rate)
    print('Saved noisy voice')
    voice_long = prob_voice.reshape(1, nb_samples * frame_length)
    sf.write(path_save_sound + 'voice_long.wav', voice_long[0, :], sample_rate)
    print('Saved voice long')
    noise_long = prob_noise.reshape(1, nb_samples * frame_length)
    sf.write(path_save_sound + 'noise_long.wav', noise_long[0, :], sample_rate)
    print('Saved noise long')
    #size of input
    dim_square_spec = int(n_fft / 2) + 1
    
    number_files = 10
    len_file = int(nb_samples/number_files)

    for i in range(number_files):
      m_amp_db_voice, _ = numpy_audio_to_matrix_spectrogram(
          prob_voice[len_file*i : len_file*(i+1)], dim_square_spec, n_fft, hop_length_fft)
      print('Voice audio to matrix ', i)
      m_amp_db_noise, _ = numpy_audio_to_matrix_spectrogram(
          prob_noise[len_file*i : len_file*(i+1)], dim_square_spec, n_fft, hop_length_fft)
      print('Noise audio to matrix ', i)
      m_amp_db_noisy_voice,  _ = numpy_audio_to_matrix_spectrogram(
          prob_noisy_voice[len_file*i : len_file*(i+1)], dim_square_spec, n_fft, hop_length_fft)
      print('Noisy voice to matrix ', i)
     
      np.save(path_save_spectrogram + 'voice_amp_db_{}'.format(i), m_amp_db_voice)
      np.save(path_save_spectrogram + 'noisy_voice_amp_db_{}'.format(i), m_amp_db_noisy_voice)

      print('-'*10)