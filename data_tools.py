from numpy import lib
import librosa 
import numpy as np 
import os 

def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    sequence_sample_length = sound_data.shape[0]
    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    list_sound_array = []
    count = 0
    for file in list_audio_files:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        
        total_duration = librosa.get_duration(y=y, sr=sr)
        if(total_duration >= min_duration):
            list_sound_array.append(audio_to_audio_frame_stack(
                y, frame_length, hop_length_frame))
        else:
            print(f"The following file {os.path.join(audio_dir,file)} is below the min duration")
        count +=1
        if count%100 == 0:
            print(count)
    return np.vstack(list_sound_array)

def blend_noise_randomly(voice, noise, nb_samples, frame_length):
    prob_voice = np.zeros((nb_samples, frame_length))
    prob_noise = np.zeros((nb_samples, frame_length))
    prob_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.7)
        prob_voice[i, :] = voice[id_voice, :]
        prob_noise[i, :] = level_noise * noise[id_noise, :]
        prob_noisy_voice[i, :] = prob_voice[i, :] + prob_noise[i, :]
    
    return prob_voice, prob_noise, prob_noisy_voice

def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)   
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)   
    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max
    )
    return stftaudio_magnitude_db, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    nb_audio = numpy_audio.shape[0]
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i]
        )
    return m_mag_db, m_phase

def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)

    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct

def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft):
    list_audio = []

    nb_spec = m_mag_db.shape[0]
    for i in range(nb_spec):
        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)
    return np.vstack(list_audio)

def scale_in(matrix_spec):
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec

def scale_ou(matrix_spec):
    matrix_spec = (matrix_spec - 6)/82
    return matrix_spec

def inv_scaled_in(matrix_spec):
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec