import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import numpy as np
import glob

#Set seed for reproducibility.
np.random.seed(42)

def gibbs_step(count, k, xk):
    '''
    Runs a single gibbs step. The visible values are initialized to xk
    '''
    W = tf.get_default_graph().get_tensor_by_name("W:0")
    bh = tf.get_default_graph().get_tensor_by_name("bh:0")
    bv = tf.get_default_graph().get_tensor_by_name("bv:0")
    hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
    xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
    return count + 1, k, xk

def gibbs_sample(k):
    '''
    Runs a k-step gibbs chain to sample from the probability distribution
    of the RBM defined by parameters W, bh, bv.
    '''
    ct = tf.constant(0)
    x = tf.get_default_graph().get_tensor_by_name("x:0")
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter,
            *args: count < num_iter, gibbs_step, [ct, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def sample(probs):
    '''
    Takes a vector of probabilities and returns random vector of 0s and 1s
    sampled from the input vector.
    '''
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def save_model(sess, saver, modelfile):
    '''
    Save model.
    '''
    saved_file = saver.save(sess, modelfile)
    print("Model saved: " + saved_file + "\n\n")

def initialize_tf(nvisible, nhidden, lrnrate, ntimestp):
    '''
    Initialize Tensorflow model.
    '''
    #Initialize variables.
    lr = tf.constant(lrnrate, name = "lr")
    ntimestp = tf.constant(ntimestp, name = "ntimestp")
    x = tf.placeholder(tf.float32, [None, nvisible], name="x")
    W = tf.Variable(tf.random_normal([nvisible, nhidden], 0.01), name="W")
    bh = tf.Variable(tf.zeros([1, nhidden], tf.float32, name="bh"))
    bv = tf.Variable(tf.zeros([1, nvisible], tf.float32, name="bv"))
    
    #Contrastive divergence algorithm.
    #Get samples of x and h from the probability distribution
    x_sample = gibbs_sample(1)
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))
    
    #Update the values of W, bh, and bv based on the  random samples.
    size_bt = tf.cast(tf.shape(x)[0], tf.float32)
    W_adder = tf.multiply(lr / size_bt,
                          tf.subtract(tf.matmul(tf.transpose(x), h),
                          tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_adder = tf.multiply(lr / size_bt,
            tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    bh_adder = tf.multiply(lr / size_bt,
            tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
    
    #Run all 3 update steps on each session update.
    updt = [W.assign_add(W_adder), bv.assign_add(bv_adder),
            bh.assign_add(bh_adder)]
    return updt
