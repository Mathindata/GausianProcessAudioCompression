"""
Provisional plan

- [DONE] any kernel: start by one implemented to gain time: Matern
- [DONE] check predict
- [DONE] check gradients
- [DONE] check fit
- more complex architecture discussed with Matin
- learn several kernel at the same time

1. the training doesn't seem stable, every time I run it the result cahnges a lot, but that might be fixed training on a large data
2. best results now with small polynomials, this can be corrected scaling down the factors for monomials with high exponent

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from scipy.io import wavfile
import scipy.signal as sps


# this function computes all the distances between two vectors of different
# length (m and n). The output is a matrix of mxn
def distance(x, x_):
    shape_x_ = tf.shape(x_)[1]
    shape_x = tf.shape(x)[1]
    batch = tf.shape(x)[0]
    xe = tf.expand_dims(x, axis=2)
    repeated_x = tf.tile(xe, [1, 1, shape_x_])
    # repeated_x = tf.transpose(repeated_x)
    xe_ = tf.expand_dims(x_, axis=1)
    subtraction = repeated_x - xe_
    return subtraction


def MaternKernel(x, x_):
    # input:  2 vector of different size (x.shape = (batch,m), x_.shape = (batch,n))
    # output: one matrix/tensor of size (batch x m x n), the kernel
    
    # matrix of all the differences
    d = tf.abs(distance(x, x_))
    
    # a is the positive multiplicative factor inside the exponential
    a = tf.square(tf.Variable(3.))  # squared to make sure it stays positive
    exponential = tf.exp(-a * d)
    
    # b is the list of coefficients of the polynomial 
    poly_degree = 6
    b = [tf.Variable(2.) for _ in range(poly_degree)]
    
    # polyval computes the polinomial of d with coefficients b
    polinomial = tf.math.polyval(b, d)
    
    kernel = exponential * polinomial
    return kernel


def testMaternKernel():
    x = tf.constant([[1., 2., 3., 4.], [1., 2., 3., 1.]])
    x_ = tf.constant([[1., 2., 2.], [1., 0., 0]])
    
    d = MaternKernel(x, x_)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    list = sess.run([x, x_, d])
    
    print([tensor.shape for tensor in list])
    
    for tensor in list:
        print('')
        print(tensor[0])

    
# A(f) = K_star*K_inv*f
def A(f):
    # this definition is flexible to inputs of different lengths
    max_senlen = tf.shape(f)[1]
    batch_size = tf.shape(f)[0]
    
    # x = [-1, -2, ... - max_senlen]
    # x_star = [0]
    x = tf.range(-1, -max_senlen - 1, -1)
    x = tf.cast(x, tf.float32)              # - they are integers, so I need to make them real for the coming operations, otherwise tf complains
    x = tf.expand_dims(x, axis=0)           # - I need to create a batch dimension, to have x for each sample in the batch
    x = tf.tile(x, [batch_size, 1])         # - repeat x for every sample in the batch
    x_star = tf.constant([0.])              # - same as before but for x_star
    x_star = tf.expand_dims(x_star, axis=0)
    x_star = tf.tile(x_star, [batch_size, 1])
    
    # this current definition is initializing different weights for
    # K_inv and K_star, but it will be easy to correct once it is in
    # a keras layer
    K_ = MaternKernel(x, x)    
    K_inv = tf.linalg.inv(K_)    
    K_star = MaternKernel(x_star, x)
    
    KK = tf.matmul(K_star, K_inv)
    f_ = tf.expand_dims(f, axis=1)
    # FIXME: the operations might not have been perfect, double check
    f_star = tf.matmul(f_, tf.transpose(KK, [0, 2, 1]))
    f_star = tf.squeeze(f_star, [2])
    return f_star
    

def testA():
    # define a dummy f to test our A(f) definition
    f = tf.constant([[1., 2., 3., 2.], [1., 2., 0, 0]])
    f_star = A(f)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    list_t = sess.run(f_star)
    
    # check if what comes out of A(f) makes sense, in terms of shapes
    # and values
    print([t.shape for t in list_t])
    for t  in list_t:
        print('')
        print(t)

# I started doing next time step prediction with GP, here I did upsampling
# A(f) = K_star*K_inv*f
def A_upsampling(f):
    stride = 2
    # this definition is flexible to inputs of different lengths
    max_senlen = tf.shape(f)[1]
    batch_size = tf.shape(f)[0]
    
    # x = [-1, -2, ... - max_senlen]
    # x_star = [0]
    x = tf.range(0, -stride * max_senlen, -stride)
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis=0)
    x = tf.tile(x, [batch_size, 1])
    
    x_star = tf.range(0, -stride * max_senlen, -1)
    x_star = tf.cast(x_star, tf.float32)
    x_star = tf.expand_dims(x_star, axis=0)
    x_star = tf.tile(x_star, [batch_size, 1])
    
    # this current definition is initializing different weights for
    # K_inv and K_star, but it will be easy to correct once it is in
    # a keras layer
    K_ = MaternKernel(x, x)    
    K_inv = tf.linalg.inv(K_)    
    K_star = MaternKernel(x_star, x)
    
    KK = tf.matmul(K_star, K_inv)
    f_ = tf.expand_dims(f, axis=1)
    # FIXME: the operations might not have been perfect, double check
    f_star = tf.matmul(f_, tf.transpose(KK, [0, 2, 1]))
    f_star = tf.squeeze(f_star, [1])
    return x, x_star, f_star


def testA_Upsampling():
    f = tf.constant([[1., 2., 3., 2.], [1., 2., 0, 0]])
    f_star = A_upsampling(f)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    list_t = sess.run(f_star)
    
    print([t.shape for t in list_t])
    for t  in list_t:
        print('')
        print(t)

    
# this is the structure of a keras Layer, 
# - the Layer inside the parenthesis is itself a class, a keras class
# that has a specific set of methods that keras recognises, and if you want
# to build your custom layer, you have to follow that structure so keras
# is able to understand that layer. Basically you need a __init__, build,
# call and compute_output_shape methods.
class GP_Upsampling(Layer):

    def __init__(self, stride, poly_degree=3, **kwargs):
        """ 
        GP Upsampling performs Gaussian Process intrapolation of the
        input signal with a Matern kernel        
        """
        # the following line allows to pass the extra arguments in **kwargs,
        # to the Layer class, kwargs is a dictionary, and when you put ** in keras
        # in front of a dictionary it gives back all the elements of it. An argument that
        # is typically inside there is the name argument, the name that you want
        # to give to your layer
        super(GP_Upsampling, self).__init__(**kwargs)
        
        # I define the variables given in the initialization (__init__)
        self.stride = stride
        self.poly_degree = poly_degree
    
    
    def build(self, input_shape):        
        # the build method of a keras Layer is where you want to initialize the variable if you
        # don't want keras to get confused.
        
        # exp_coef is the positive multiplicative factor inside the exponential
        self.exp_coef = self.add_weight(name='exponent_coeff',
                                 shape=(),
                                 initializer='uniform',
                                 trainable=True)
        
        # poly_coef is the list of coefficients of the polynomial 
        self.poly_coef = [self.add_weight(name='poly_{}'.format(i),
                                  shape=(),
                                  initializer='uniform',
                                  trainable=True) 
                                  for i in range(self.poly_degree)]

        super(GP_Upsampling, self).build(input_shape) 
    
    def Matern(self, x, x_):
        # matrix of all the differences
        d = tf.abs(distance(x, x_))
        
        exponential = tf.exp(-self.exp_coef * d)
        
        # polyval computes the polinomial of d with coefficients poly_coef
        polinomial = tf.math.polyval(self.poly_coef, d)
        
        kernel = exponential * polinomial
        return kernel

    def call(self, f):
        # this definition is flexible to inputs of different lengths
        max_senlen = tf.shape(f)[1]
        batch_size = tf.shape(f)[0]
        
        # x = [-0, -2, -4]
        # x_star = [-0, -1, -2, -3, -4, -5]
        x = tf.range(0, -self.stride * max_senlen, -self.stride)
        x = tf.cast(x, tf.float32)
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, [batch_size, 1])
        
        x_star = tf.range(0, -self.stride * max_senlen, -1)
        x_star = tf.cast(x_star, tf.float32)
        x_star = tf.expand_dims(x_star, axis=0)
        x_star = tf.tile(x_star, [batch_size, 1])
        
        # defining Matern as a method of the Layer allows me to call 
        # the operation with the weights updated through training
        K_ = self.Matern(x, x)    
        K_inv = tf.linalg.inv(K_)    
        K_star = self.Matern(x_star, x)
        
        KK = tf.matmul(K_star, K_inv)
        f_ = tf.expand_dims(f, axis=1)
        
        f_star = tf.matmul(f_, tf.transpose(KK, [0, 2, 1]))
        f_star = tf.squeeze(f_star, [1])
        return f_star
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*self.stride)

    
    
def testLayer():
    data_path = 'data/file_example_WAV_1MG.wav'

    stride = 6
    sampling_rate, data = wavfile.read(data_path)
    data = data[:1002].T/np.max(data)
    
    undersampled_data = data[:, ::stride]    
    
    print(data.shape)
    print(undersampled_data.shape)
    
    input_signal = Input((None,))    
    gp_upsampling = GP_Upsampling(stride)(input_signal)
    model = Model(input_signal, gp_upsampling)
    
    bt_prediction = model.predict(undersampled_data)
    print(bt_prediction.shape)
    
    model.compile('adam', 'mse')
    history = model.fit(undersampled_data, data, epochs=10)

    at_prediction = model.predict(undersampled_data)
    print(at_prediction.shape)
    
    
    
    plt.plot(data[0])
    plt.plot(bt_prediction[0], label='before training')
    plt.plot(at_prediction[0], label='after training')
    plt.axis([0, data.shape[1], -.3, .3])
    plt.legend(loc='upper left')
    plt.savefig('upsampling.pdf')
    
if __name__ == '__main__':
    # testMaternKernel()
    # testA()
    #testA_Upsampling()
    testLayer()
    
