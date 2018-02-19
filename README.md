# conv_lstm_cell

Custom built ConvLSTM cell in Tensorflow and Pytorch

Models inspired by [Convolutional LSTM](https://arxiv.org/pdf/1506.04214.pdf)

* pt_conv2dlstm_cell.py
```
PyTorch implementation of ConvLSTM initializes weights and defines forward method for inference
```
* tf_conv2dlstm_cell.py
```
Tensorflow implementation of ConvLSTM static graph and prepares equations for inference time
```

* pytorch_model.py
```
Defining the model for training that uses pt_conv2dlstm_cell input normally image
```
* tensorflow_model.py
```
Defines static graph for training that uses tf_conv2dlstm_cell
```
