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




Suggestion:
    Use tanh over ReLu

    
