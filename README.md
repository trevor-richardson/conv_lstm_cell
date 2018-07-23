# conv_lstm_cell

Custom built ConvLSTM cell in Tensorflow and Pytorch <br/>

Models inspired [Convolutional LSTM](https://arxiv.org/pdf/1506.04214.pdf)

<img src="https://github.com/trevor-richardson/conv_lstm_cell/blob/master/math/convlstm.png" width="750">


PyTorch implementation of ConvLSTM initializes weights and defines forward method for inference
```
see pt_conv2dlstm_cell.py
```
<br/>

Tensorflow implementation of ConvLSTM static graph and prepares equations for inference time
```
see tf_conv2dlstm_cell.py
```
<br/>

Defining the model for training that uses pt_conv2dlstm_cell input normally image
```
see pytorch_model.py
```
<br/>

Defines static graph for training that uses tf_conv2dlstm_cell
```
see tensorflow_model.py
```
