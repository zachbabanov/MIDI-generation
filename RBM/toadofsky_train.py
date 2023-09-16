import tensorflow.compat.v1 as tf

import numpy as np
import midi_utils as mu
import tf_utils as tf_u
from tqdm import tqdm
import mido
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


#Set seed for reproducibility.
tf.set_random_seed(42)

#Supress TF warnings for clean printed output.
tf.logging.set_verbosity(tf.logging.ERROR)

#Set variables.
NTIMESTP = 15                   #Number of MIDI time steps to train at a time
NOTESPAN = 128                  #Span all standard MIDI notes.
NVISIBLE = NOTESPAN * NTIMESTP  #Number of visible nodes.
NHIDDEN = 50                    #Number of hidden nodes
NEPOCHS = 200                   #Number of epochs
BATCHSIZE = 100                 #Batch size
LRNRATE = 0.005                 #Learning rate
MIDIDIR = "./vgmusic/"          #Directory containing MIDI files to train on
MODELDIR = "./models/"          #Directory where trained models are saved
MODELNAME = "toadofsky"         #Name of model

#Process MIDI files into binary training and validation data.
train_songs = mu.load_midis(MIDIDIR)
print("Midis loaded for training: " + str(len(train_songs)))

'''
INITIALIZATION OF MODEL
'''

#Initialize Tensorflow model.
updt = tf_u.initialize_tf(NVISIBLE, NHIDDEN, LRNRATE, NTIMESTP)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
TRAINING LOOP
'''

print("\n\nTraining...\n\n")

#Train on data set NEPOCHS number of times.
for epoch in tqdm(range(NEPOCHS)):

    #Cycle through each song in training set.
    for song in train_songs:
        
        #Reshape matrix to fit BATCHSIZE
        song = song[:int(np.floor(song.shape[0] // NTIMESTP) * NTIMESTP)]
        song = np.reshape(song,
                [song.shape[0] // NTIMESTP, song.shape[1] * NTIMESTP])

        #Train the RBM on BATCHSIZE examples at a time
        for i in range(1, len(song), BATCHSIZE):
            tr_x = song[i:i + BATCHSIZE]
            feed_dict = {"x:0": tr_x}
            sess.run(updt, feed_dict=feed_dict)

'''
Save Model
'''

print("\n\nTraining complete!")
tf_u.save_model(sess, saver, MODELDIR + MODELNAME)
print("Model saved! (" + MODELDIR + MODELNAME + ".meta)")
