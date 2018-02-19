# conv_lstm_cell

Custom built ConvLSTM cell in Tensorflow and Pytorch

Models inspired by [Convolutional LSTM](https://arxiv.org/pdf/1506.04214.pdf)

* conv2dlstm_cell.py
```
Initializes the weights to be used and the forward() method which performs ConvLSTM equations at inference time
```
* conv2dlstm_cell.py
```
Initializes static graph and prepares a cyclic graph and performces ConvLSTM equations at inference time
```

* pytorch_model.py
```
Defining the model for training that uses pt_conv2dlstm_cell input normally image
```
* tensorflow_model.py
```
Defines static graph for training that uses tf_conv2dlstm_cell
```
