import os
from subprocess import call
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.feature.inverse as lfi

from scipy.io.wavfile import write


# %% generating wav files from videos (timestamped with transcriptions)
def generate_wav_files():
    wav_files = []
    count = 0
    for filename in glob.glob("./data/captions_vids/*.wav"):
        if count == 5:
            break
        print(filename)
        y, sr = librosa.load(filename, sr=None)
        wav_sr = dict([('signal', y), ('sr', sr), ('caption_file', filename)])
        wav_files.append(wav_sr)
        count += 1

    wav_files = pd.DataFrame(wav_files)

    print(wav_files.head())
    return wav_files


#%% function to chop 
def chop(wav_file, n_fft, hop_length, start_end=None):
    '''

    :param wav_file: audio file
    :param caption_csv: csv readable into dataframe divided into time chunks with [start, end] times and captions for each chunk
    :param sample_rate: sample rate of wav file (samples per second)
    :param start_end: timestamp tuple in seconds
    :return:
    '''

    signal = wav_file['signal']
    sample_rate = wav_file['sr']
    captions = wav_file['caption_file']

    start = start_end[0] * sample_rate
    end = start_end[1] * sample_rate

    #length of windowed signal after padding with zeroes -> 1024 corresponds to a physical duration of 23ms at sr=22050
    n_fft = n_fft

    #number of audio samples between adjacent STFT columns: defaults to n_fft // 4
    hop_length = hop_length

    #generate waveplot
    librosa.display.waveplot(signal[start:end], sr=sample_rate)
    plt.show()

    #create spectrogram
    stft = np.abs(librosa.stft(signal[start:end], n_fft=n_fft, hop_length=hop_length))
    mel = librosa.feature.melspectrogram(sr=sample_rate, S=stft ** 2)
    log_mel = librosa.amplitude_to_db(mel)

    librosa.display.specshow(log_mel, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    print("generated using n_fft:", n_fft, " and hop_length: ", hop_length)
    plt.show()
    return stft, mel, log_mel


#%% generate wav_files
wav_files = generate_wav_files()

#%% generate mel spectrogram
first_vid = wav_files.iloc[0]
n_fft = 2048
hop_length = 512
stft, mel, log_mel = chop(first_vid, n_fft=n_fft, hop_length=hop_length, start_end=(90, 100))


#%% convert mel spectrogram back to audio using librosa
sr = 48000
res = lfi.mel_to_audio(mel, sr=sr, n_fft=2048, hop_length=512)


write("test2.wav", sr, res.astype(np.float32))

#%% play sound file

def play(wav_file):
    call(['aplay', wav_file])

#%%
play('test2.wav')