
import tensorflow as tf
import numpy as np
import sys
'''
The following is an object which implements a stateful ConvLSTM2D by default I include the cell state in calculating input, forget and output gates. I found that
using hard sigmoid for the non-linear activation of output forget and input gate improved the intelligibility of my activations (reduced the fuzziness of visualizing activations)
By default, I set kernel weights with xavier glorot initialization and recurrent weights with orthogonal initialization; however, you are free to set the
kernel and recurrent weights to whatever you want as long as your function that you pass to this object for initializing is of the form f('name', shape).
Passing functions for weights, recurrent weights and biases require that the function has the following parameter in the following order (name, shape)
This allows for the reset state to be a hyperparameter in which you can learn the most generalizable model based on how long the recurrency analysis should be
Also allows for the reset states never to be used in which case
'''

class StatefulConvLSTM2D(object):
    def __init__(self, input_shape, no_filters, kernel_shape, strides, layer_no, weight_init=None, padding='SAME', reccurent_weight_init=None, cell_weight_init=None, bias_init=None, drop=None, rec_drop=None, use_cell=True):
        if(weight_init==None):
            #weights should be of the shape (filter_width, filter_height, num_prev_channels, no_filters)
            self.W_f = xavier_uniform_init("W_f_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_i = xavier_uniform_init("W_i_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_o = xavier_uniform_init("W_o_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_c = xavier_uniform_init("W_c_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
        else:
            self.W_f = weight_init("W_f_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_i = weight_init("W_i_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_o = weight_init("W_o_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])
            self.W_c = weight_init("W_c_" + str(layer_no), [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters])

        if(reccurent_weight_init == None):
            #Weight matrices for U should be convolved and a 4d tensorf of shape (kernel_shape[0], kernel_shape[0], no_filters, no_filters)
            self.U_f = initialize_orthogonal_weights("U_f_" + str(layer_no),[kernel_shape[0], kernel_shape[1], no_filters, no_filters] )
            self.U_i = initialize_orthogonal_weights("U_i_" + str(layer_no),[kernel_shape[0], kernel_shape[1], no_filters, no_filters] )
            self.U_o = initialize_orthogonal_weights("U_o_" + str(layer_no),[kernel_shape[0], kernel_shape[1], no_filters, no_filters] )
            self.U_c = initialize_orthogonal_weights("U_c_" + str(layer_no),[kernel_shape[0], kernel_shape[1], no_filters, no_filters] )
        else:
            self.U_f = reccurent_weight_init("U_f" + str(layer_no), [kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters])
            self.U_i = reccurent_weight_init("U_i" + str(layer_no), [kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters])
            self.U_o = reccurent_weight_init("U_o" + str(layer_no), [kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters])
            self.U_c = reccurent_weight_init("U_c" + str(layer_no), [kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters])

        if(cell_weight_init==None):
            if (padding == 'SAME'):
                self.V_f = recurrent_weight_initializer("V_f_" + str(layer_no),[int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
                self.V_i = recurrent_weight_initializer("V_i_" + str(layer_no),[int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
                self.V_o = recurrent_weight_initializer("V_o_" + str(layer_no),[int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
            else:
                self.V_f = recurrent_weight_initializer("V_f_" + str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
                self.V_i = recurrent_weight_initializer("V_i_" + str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
                self.V_o = recurrent_weight_initializer("V_o_" + str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
        else:
            if (padding == 'SAME'):
                self.V_f = cell_weight_init("V_f" + str(layer_no), [int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
                self.V_i = cell_weight_init("V_i" + str(layer_no), [int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
                self.V_o = cell_weight_init("V_o" + str(layer_no), [int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters] )
            else:
                self.V_f = cell_weight_init("V_f" + str(layer_no), str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
                self.V_i = cell_weight_init("V_i" + str(layer_no), str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
                self.V_o = cell_weight_init("V_o" + str(layer_no), str(layer_no),[int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters] )
        if(bias_init==None):
            #biases should be of the shape (no_filters)
            self.b_f = tf.Variable(tf.zeros(shape = [no_filters]), dtype=tf.float32, name="b_f_" + str(layer_no))
            self.b_i = tf.Variable(tf.zeros(shape = [no_filters]), dtype=tf.float32, name="b_i_" + str(layer_no))
            self.b_o = tf.Variable(tf.zeros(shape = [no_filters]), dtype=tf.float32, name="b_o_" + str(layer_no))
            self.b_c = tf.Variable(tf.zeros(shape = [no_filters]), dtype=tf.float32, name="b_c_" + str(layer_no))
        else:
            #biases
            self.b_f = bias_init("b_" + str(layer_no), [no_filters])
            self.b_i = bias_init("b_" + str(layer_no), [no_filters])
            self.b_o = bias_init("b_" + str(layer_no), [no_filters])
            self.b_c = bias_init("b_" + str(layer_no), [no_filters])

        self.stride = strides
        self.kernel = kernel_shape
        self.no_filters = no_filters
        self.inp_shape = input_shape
        self.use_cell_for_gates = use_cell
        self.pad = padding
        #The following sets my dropout
        if(drop==None):
            self.dropout = 0
        else:
            self.dropout = min(1., max(0., drop))

        if(rec_drop == None):
            self.rec_dropout = 0
        else:
            self.rec_dropout = min(1., max(0., rec_drop))
        #exit program and send error message if shapes of matrices are incorrect or self.V_f.get_shape() != (int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters)
        if (self.pad == 'SAME'):
            if(self.W_f.get_shape() != (kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters) or self.U_f.get_shape() != (kernel_shape[0], kernel_shape[1], no_filters, no_filters)
                or self.V_f.get_shape() != (int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters) or self.b_f.get_shape() != (no_filters)):
                print("Dimensions for weight_init return should be (input_dimension, num_hidden_neurons)\nDimensions for reccurent_weight_init shoudl be (num_hidden_neurons, num_hidden_neurons)")
                print("Dimensions for bias_init (num_hidden_neurons)")
                print(self.W_f.get_shape(), "Current weight_init shape           ---- The shape should be ", (kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
                print(self.U_f.get_shape(), "Current reccurent_weight_init shape ---- The shape should be ", (kernel_shape[0], kernel_shape[1], no_filters, no_filters))
                print(self.V_f.get_shape(), "Current cell_weight_init shape      ---- The shape should be ", (int( int(input_shape[1]) / strides[1]), int( int(input_shape[2]) / strides[2]), no_filters))
                print(self.b_f.get_shape(), "Current bias shape                  ---- The shape should be ", (no_filters,))
                sys.exit()
        else:
            if(self.W_f.get_shape() != (kernel_shape[0], kernel_shape[1], input_shape[-1], no_filters) or self.U_f.get_shape() != (kernel_shape[0], kernel_shape[1], no_filters, no_filters)
                or self.V_f.get_shape() != (int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters) or self.b_f.get_shape() != (no_filters)):
                print("Dimensions for weight_init return should be (input_dimension, num_hidden_neurons)\nDimensions for reccurent_weight_init shoudl be (num_hidden_neurons, num_hidden_neurons)")
                print("Dimensions for bias_init (num_hidden_neurons)")
                print(self.W_f.get_shape(), "Current weight_init shape           ---- The shape should be ", (kernel_shape[0], kernel_shape[1], int(input_shape[-1]), no_filters))
                print(self.U_f.get_shape(), "Current reccurent_weight_init shape ---- The shape should be ", (kernel_shape[0], kernel_shape[1], no_filters, no_filters))
                print(self.V_f.get_shape(), "Current cell_weight_init shape      ---- The shape should be ", (int( int(input_shape[1] - kernel_shape[0] + 1) / strides[1]), int( int(input_shape[2] - kernel_shape[1] + 1) / strides[2]), no_filters))
                print(self.b_f.get_shape(), "Current bias shape                  ---- The shape should be ", (no_filters,))
                sys.exit()

    def execute_lstm(self, X_t, previous_hidden_memory_tuple):
        h_t_previous, c_t_previous = tf.unstack(previous_hidden_memory_tuple, axis=1)
        #apply dropout
        if self.dropout != 0:
            X_t = tf.nn.dropout(X_t, 1-self.dropout)
        if self.rec_dropout !=0:
            h_t_previous = tf.nn.dropout(h_t_previous, 1-self.rec_dropout)
            c_t_previous = tf.nn.dropout(c_t_previous, 1-self.rec_dropout)

        if self.use_cell_for_gates:
            #f(t) = sigmoid(W_f (conv) x(t) + U_f (conv) h(t-1) + V_f (*) c(t-1)  + b_f)
            f_t = hard_sigmoid(
                conv2d(X_t, self.W_f, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_f, [1, 1, 1, 1]) + c_t_previous * self.V_f + self.b_f #w_f needs to be the previous input shape by the number of hidden neurons
            )
            #i(t) = sigmoid(W_i (conv) x(t) + U_i (conv) h(t-1) + V_i (*) c(t-1)  + b_i)
            i_t = hard_sigmoid(
                conv2d(X_t, self.W_i, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_i, [1, 1, 1, 1]) + c_t_previous * self.V_i + self.b_i
            )
            #o(t) = sigmoid(W_o (conv) x(t) + U_o (conv) h(t-1) + V_i (*) c(t-1) + b_o)
            o_t = hard_sigmoid(
                conv2d(X_t, self.W_o, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_o, [1, 1, 1, 1]) + c_t_previous * self.V_o + self.b_o
            )
        else:
            f_t = hard_sigmoid(
                conv2d(X_t, self.W_f, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_f, [1, 1, 1, 1]) + self.b_f #w_f needs to be the previous input shape by the number of hidden neurons
            )
            #i(t) = sigmoid(W_i (conv) x(t) + U_i (conv) h(t-1) + V_i (*) c(t-1)  + b_i)
            i_t = hard_sigmoid(
                conv2d(X_t, self.W_i, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_i, [1, 1, 1, 1]) + self.b_i
            )
            #o(t) = sigmoid(W_o (conv) x(t) + U_o (conv) h(t-1) + V_i (*) c(t-1) + b_o)
            o_t = hard_sigmoid(
                conv2d(X_t, self.W_o, self.stride, pad=self.pad) + conv2d(h_t_previous, self.U_o, [1, 1, 1, 1]) + self.b_o
            )
        #c(t) = f(t) (*) c(t-1) + i(t) (*) hypertan(W_c (conv) x_t + U_c (conv) h(t-1) + b_c)
        c_hat_t = tf.nn.tanh(
            conv2d(X_t, self.W_c, self.stride) + conv2d(h_t_previous, self.U_c, [1, 1, 1, 1]) + self.b_c
        )
        c_t = (f_t * c_t_previous) + (i_t * c_hat_t)
        #h_t = o_t * tanh(c_t)
        h_t = o_t * tf.nn.tanh(c_t)
        #h(t) = o(t) (*) hypertan(c(t))
        next_mem = tf.stack([h_t, c_t], axis=1)
        return next_mem

'''
The following are some helper functions for default initialization purposes
Initialize recurrent weight matrices with orthogonal initizlization
Initialize kernel weights with xavier uniform initialization
'''
def conv2d(x, W, stride, pad='SAME'):
  return tf.nn.conv2d(x, W, strides=stride, padding=pad)

def initialize_orthogonal_weights(name, shape):
    """recurr_weight_variable generates a recurrent weight variable of a given shape."""
    X = np.random.normal(0.0, 1.0, (shape[0], shape[1]*shape[2]*shape[3]))
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    initial = Vt.reshape((shape[0], shape[1], shape[2], shape[3]))
    return tf.Variable(initial, name=name,dtype=tf.float32)

def xavier_uniform_init(name, shape):
    fan_in = float(shape[2])
    fan_out = float(shape[3])
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation (keras uses 1)
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    initial = tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)
    return tf.Variable(initial, name=name,dtype=tf.float32)

def recurrent_weight_initializer(name, shape):
    X = np.random.normal(0.0, 1.0, (shape[0], shape[1]*shape[2]))
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    initial = Vt.reshape((shape[0], shape[1], shape[2]))
    return tf.Variable(initial, name=name,dtype=tf.float32)

def hard_sigmoid(x):
    """hard sigmoid for convlstm"""
    x = (0.2 * x) + 0.5
    zero = tf.convert_to_tensor(0., x.dtype.base_dtype)
    one = tf.convert_to_tensor(1., x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, one)
    return x
