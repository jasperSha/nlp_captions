import glob
import pandas as pd
import librosa

def load_wav_files():
    '''
    handle loading audio data into memory
    :return:
    '''
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
