# MIDI-generation
Generating a MIDI-format piano songs with neural network on python

## Requirements
Installing the following packages using pip:
* Music21
* Keras
* Tensorflow
* h5py

## Training
To train the network you run:

```
python train.py
```
The network will use every midi file in ./midi_songs to train the network

Once you have trained the network you can generate text using **generate.py**:

```
python generate.py
```
