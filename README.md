Feed a neural net youtube videos with premade, labelled captions in order to be able to automatically caption future, unlabelled videos.


Using a variational Autoencoder. Basically we feed it some input with an accompanying loss function, saying that the input has some degree of uncertainty to it. Then the neural net learns to label outputs, but because there is accompanying noise, it's able to generalize the translation a lot better, so that when there is input that is within those loss bounds, it can also extrapolate and get the correct/similar output as the original label.

Encoder(input to hidden) ---> Decoder(hidden to output)

Steps:
    1. load audio from .wav file
    2. resample and convert to stereo(two channel)
    3. resize to fixed length
    4. audio augmentation time shift (??)
    5. convert to mel spectrogram
    6. Spectrogram Augmentation SpecAugment

7/31:
Right now trying to determine how to separate the wav files and captions. Do I feed them in as entire lines with their accompanying captions, or do I somehow split the wav files on each word with each captioned word? The issue with that is I'd have to do it manually or figure out some way of programmatically splitting each chunk of dialogue, maybe using an amplitude pass to cut on silence, then match to the labeled caption data?

An easier way is to use datasets that are already specifically split on words, and also provide the extra benefit of simulating environmental noise as "variance" to help prevent overfitting.

8/1
Solution to word splitting issue: apply sequence modeling using Connectionist Temporal Classification (CTC), which, when fed a sequence of tokens from an RNN(effective for its contextual distribution, but any NN that produces fixed-size input slice -> distribution over output classes/tokens will work), can compute a probability distribution of tokens and then collapse them. The LibriSpeech dataset works well with this as it's already split into digestible chunks of waveforms + captions, although Youtube datasets will work as well.

8/8
http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Chen_16.pdf
pre-trained cnn + transformer for audio captioning
    - they used 64-band log mel
    - amplitude -> decibel scale
    - sampling_rate=44100, n_fft=1024, num_samples between frames=512
    - zero-padded in batch -> tensor
    - captions pre-processed
        - tokenized
        - punctuation removed
        - all letters lowercased
        - start and end tokens to demarcate sentences
    - caption vocabulary preprocessed
        - cnn pretrained on classification task using 300 word classes
        - removed articles, found words with "-ing", "-ly", etc, removed the postfixes, added to frequency of stem words
        - 300 words with highest frequency selected as classes
    - labelled audio using multi-hot vector
    - input is the log mel spectrogram of audio
    - output is f(x) in [0, 1]^K, representing probability of K classes
    - binary cross-entropy loss
    - multi-label classification and encoder of audio captioning use same CNN -> easier transfer learning
    - 60/20/20 train, valid, test sets
    - batch=16, lr=3e-4, l2 regularization, lambda=1e-6, label smoothing epsilon=1e-1, dropout p=0.2

Suggestion:
    Use tanh over ReLu

    
