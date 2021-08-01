import subprocess
def play(sound_file):
    '''
    playback of flac/wav sound files
    :param sound_file:
    :return:
    '''
    extension = sound_file.split('.')[-1]
    if extension == 'wav':
        subprocess.Popen(['aplay', sound_file])
    if extension == 'flac':
        flac = subprocess.Popen(('flac', '-c', '-d', sound_file), stdout=subprocess.PIPE)
        out = subprocess.check_output(('aplay'), stdin=flac.stdout)
        flac.wait()
    else:
        print('unhandled filetype; extension is: ', extension)
