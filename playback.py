import subprocess
import platform
import sox


def play(sound_file):
    '''
    playback of flac/wav sound files
    :param sound_file:
    :return:
    '''
    current_os = platform.system()

    # alsa on ubuntu -> aplay; sox on mac -> play
    if current_os == 'Darwin':
        os_cmd = 'play'
    elif current_os == 'Linux':
        os_cmd = 'aplay'
    else:
        print("unknown OS, can't play audio in terminal!")
        return

    extension = sound_file.split('.')[-1]
    if extension == 'wav':
        print('running wav file')
        subprocess.Popen([os_cmd, sound_file])
    elif extension == 'flac' and current_os == 'Darwin':
        print('running flac file on mac architecture')
        subprocess.call([os_cmd, sound_file])
    elif extension == 'flac' and current_os == 'Linux':
        print('running flac file on linux architecture')
        flac = subprocess.Popen(('flac', '-c', '-d', sound_file), stdout=subprocess.PIPE)
        out = subprocess.check_output((os_cmd), stdin=flac.stdout)
        flac.wait()
    else:
        print('unhandled filetype; extension is: ', extension)
