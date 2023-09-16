
import numpy as np
import midi_utils as mu
import tf_utils as tf_u
import mido
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
tf.disable_v2_behavior()


#Supress TF warnings for clean printed output.

#Set variables.
INSTRUMENT = 114            #General MIDI instrument number
NMEASURES = 32              #Number of measures in final MIDI
NOTEPERMEAS = 10            #Number of possible notes per measure
TIMESCALE = 16              #Scale from encoded moments to MIDI ticks
MODELDIR = "./models/"      #Directory containing trained models
MODELNAME = "toadofsky"     #Name of model without file extensions
OUTDIR = "./test_output/"   #Output directory

'''
INITIALIZATION OF MODEL
'''

#Initialize Tensorflow model.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Load specified model.
saver = tf.train.import_meta_graph(MODELDIR + MODELNAME + ".meta")
saver.restore(sess, MODELDIR + MODELNAME)
ntimestp = sess.run(sess.graph.get_tensor_by_name("ntimestp:0"))
nvisible = 128 * ntimestp

'''
MIDI GENERATION
'''

midi_out = []
for measure in range(NMEASURES):

    #Run a gibbs chain with visible nodes initialized to 0.
    feed_dict={"x:0": np.zeros((NOTEPERMEAS, nvisible))}
    gs = tf_u.gibbs_sample(1).eval(session=sess, feed_dict=feed_dict)
    
    for i in range(gs.shape[0]):
        if any(gs[i, :]):
            #Reshape measure vector and append to output MIDI.
            measure = np.reshape(gs[i, :], (ntimestp, 128))
            midi_out.extend(measure.copy())

'''
FILE OUTPUT
'''

#Convert encoded song to MIDI and save.
mu.make_midi(midi_out, INSTRUMENT, TIMESCALE, OUTDIR + MODELNAME + ".mid")
