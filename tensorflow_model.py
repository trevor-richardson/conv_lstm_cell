import tensorflow as tf
import numpy as np
import sys
import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split('robotics-research')[0]

sys.path.append(dir_path + 'robotics-research/deep_learning/tensorflow/visualizing_network')
sys.path.append(dir_path + 'robotics-research/deep_learning/tensorflow/data_generator')
sys.path.append(dir_path + 'robotics-research/deep_learning/tensorflow/stateful_lstm')
sys.path.append(dir_path + 'robotics-research/deep_learning/tensorflow/conv_lstm_stateful')


from anticipation_generator import StateImageDataGenerator #I built this so that I could load
from activation_visualizer import VisualizeActivations
from stateful_convlstm_cell import StatefulConvLSTM2D
from stateful_lstm_cell import StatefulLSTM

#this needs to accept state and image data and output perceived threat of future situations
class AnticipationModel(object):
    def __init__(self):
        print("building anticipation model")
        #file paths
        #Important training and testing parameters
        self.batch_size = 64
        self.nb_epoch = 100
        self.state_size = 7
        self.learning_rate = .0001
        self.keeping_prob = .7

        self.output_layer_neurons = 1

        #conv info
        self.kernel_size = (5,5)
        self.filters_0 = 20
        self.filters_1 = 30
        self.filters_2 = 40
        self.strides = [1, 2, 2, 1] #(batch_sz, x, y, z) -1 represents fit any batch
        self.pad = 'SAME'

        #pure lstm layers
        self.lstm_hidden_0 = 10
        self.lstm_hidden_1 = 10
        self.lstm_hidden_2 = 10

        #concat layers
        self.concat_dense_neurons = 50
        self.output_layer_neurons = 1

        #placeholders
        self.image_input = tf.placeholder(tf.float32, name="image_input", shape=[None, 64, 64, 3])
        self.state_input = tf.placeholder(tf.float32, name="state_input", shape=[None, self.state_size])
        self.y_ = tf.placeholder(tf.float32, name="ground_truth_label", shape=[None, self.output_layer_neurons])
        self.keep_prob = tf.placeholder(tf.float32)
        #The following dynamically creates the prev_state_placeholder dimension based on kernel dimension, number of filters and padding
        if self.pad == 'SAME':
            self.prev_state_0_img = tf.placeholder(tf.float32, name="previous_state_0" , shape=[None, 2, int(64 / self.strides[1]), int(64/self.strides[2]), self.filters_0])
            self.prev_state_1_img = tf.placeholder(tf.float32, name="previous_state_1" , shape=[None, 2, int(32 / self.strides[1]), int(32/self.strides[2]), self.filters_1])
            self.prev_state_2_img = tf.placeholder(tf.float32, name="previous_state_2" , shape=[None, 2, int(16 / self.strides[1]), int(16/self.strides[2]), self.filters_2])
        else:
            self.prev_state_0_img = tf.placeholder(tf.float32, name="previous_state_0" , shape=[None, 2, int(int(64 - self.kernel_size[0] + 1) / self.strides[1]), int(int(64 - self.kernel_size[1] + 1) /self.strides[2]), self.filters_0])
            self.prev_state_1_img = tf.placeholder(tf.float32, name="previous_state_1" , shape=[None, 2, int(int(32 - self.kernel_size[0] + 1) / self.strides[1]), int(int(32 - self.kernel_size[1] + 1) /self.strides[2]), self.filters_1])
            self.prev_state_2_img = tf.placeholder(tf.float32, name="previous_state_2" , shape=[None, 2, int(int(16 - self.kernel_size[0] + 1) / self.strides[1]), int(int(16 - self.kernel_size[1] + 1) /self.strides[2]), self.filters_2])
        self.prev_state_0_stt = tf.placeholder("float", [None, 2, self.lstm_hidden_0])
        self.prev_state_1_stt = tf.placeholder("float", [None, 2, self.lstm_hidden_1])
        self.prev_state_2_stt = tf.placeholder("float", [None, 2, self.lstm_hidden_2])
        #dense layer and output layer
        #data_generator
        self.load_miss_or_hit = 1
        self.stateful_reset = 70
        self.video_sz = 70

    def build_network(self):
        print("Building network")
        #CONVLSTM
        conv_lstm_0 = StatefulConvLSTM2D(self.image_input.get_shape(), self.filters_0, self.kernel_size, self.strides, 0)

        self.pred_0_img = conv_lstm_0.execute_lstm(self.image_input, self.prev_state_0_img)
        hidden_lstm_output_0 = self.pred_0_img[:, 0]

        conv_lstm_1 = StatefulConvLSTM2D(hidden_lstm_output_0.get_shape(), self.filters_1, self.kernel_size, self.strides, 1)
        self.pred_1_img = conv_lstm_1.execute_lstm(hidden_lstm_output_0, self.prev_state_1_img)
        hidden_lstm_output_1 = self.pred_1_img[:, 0]

        conv_lstm_2 = StatefulConvLSTM2D(hidden_lstm_output_1.get_shape(), self.filters_2, self.kernel_size, self.strides, 2)
        self.pred_2_img = conv_lstm_2.execute_lstm(hidden_lstm_output_1, self.prev_state_2_img)
        hidden_lstm_output_2 = self.pred_2_img[:, 0]

        #LSTM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Make sure dropout here works becauase I changed to keep prob now
        lstm_object_0 = StatefulLSTM(self.state_size, self.lstm_hidden_0, 0, drop=self.keep_prob)
        lstm_object_1 = StatefulLSTM(self.lstm_hidden_0, self.lstm_hidden_1, 1, drop=self.keep_prob)
        lstm_object_2 = StatefulLSTM(self.lstm_hidden_1, self.lstm_hidden_2, 2, drop=self.keep_prob)

        self.pred_0_state = lstm_object_0.execute_lstm(self.state_input , self.prev_state_0_stt)
        hidden_lstmstate_output_0 = self.pred_0_state[:, 0] #This shape is (batch_size, 2, num_hidden_neurons) -- the 2 represents at pred_0[0] h_t output and pred_0[1] c_t output

        self.pred_1_state = lstm_object_1.execute_lstm(hidden_lstmstate_output_0, self.prev_state_1_stt)
        hidden_lstmstate_output_1 = self.pred_1_state[:, 0]

        self.pred_2_state = lstm_object_2.execute_lstm(hidden_lstmstate_output_1, self.prev_state_2_stt)
        hidden_lstmstate_output_2 = self.pred_2_state[:, 0]

        #OUTPUT
        #concat output
        flatten_image = tf.contrib.layers.flatten(hidden_lstm_output_2)
        self.concat = tf.concat([flatten_image, hidden_lstmstate_output_2], axis=1, name='concat')
        # print(self.concat.get_shape(), "concat_shape")

        #outputlayers
        W_concat_dense = initialize_weight_xavier('concat_dense_neurons_weight', [int(self.concat.get_shape()[-1]), self.concat_dense_neurons])
        b_concat_dense = tf.zeros(self.concat_dense_neurons) #If the above batch_norm doesn't work work on the other implementation instead
        hidden_concat = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(self.concat, W_concat_dense) + b_concat_dense), self.keep_prob)

        W_out = initialize_weight_xavier('output_weight', [self.concat_dense_neurons, self.output_layer_neurons])
        b_out = tf.zeros(self.output_layer_neurons) #If the above batch_norm doesn't work work on the other implementation instead

        self.output_layer = tf.nn.sigmoid(tf.matmul(hidden_concat, W_out) + b_out)

        #LOSS FUNCTION
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.output_layer, name="Binary_Crossentropy_Loss")

        self.correct_pred = tf.equal(tf.round(self.output_layer), self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
