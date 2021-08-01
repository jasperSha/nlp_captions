import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.feature.inverse as lfi

from scipy.io.wavfile import write


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

    #some regex malarkey to get just the filename
    filename = captions.split('/')[-1].split('.')[0]

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
    return stft, mel, log_mel, sample_rate, filename


#%% generate wav_files
from load_wav import load_wav_files
wav_files = load_wav_files()

#%% generate mel spectrogram
first_vid = wav_files.iloc[0]

n_fft = 2048
hop_length = 1024
stft, mel, log_mel, sample_rate, filename = chop(first_vid, n_fft=n_fft, hop_length=hop_length, start_end=(90, 100))


#%% convert mel spectrogram back to audio using librosa
res = lfi.mel_to_audio(mel, sr=sample_rate, n_fft=2048, hop_length=1024)

write("./data/reversed/file=%s n_fft=%s hop_length=%s sr=%s.wav"%(filename, n_fft, hop_length, sample_rate), sample_rate, res.astype(np.float32))


#%% playback
from playback import play
play('./data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac')

flipped_sample = "./data/reversed/file=contra_00 n_fft=2048 hop_length=1024 sr=48000.wav"
play(flipped_sample)

