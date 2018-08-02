# conv_lstm_cell

This repository implements a custom built ConvLSTM cell in Tensorflow and Pytorch. <br/>

The model was first introduced in [Convolutional LSTM](https://arxiv.org/pdf/1506.04214.pdf).

<img src="https://github.com/trevor-richardson/conv_lstm_cell/blob/master/math/convlstm.png" width="750">


The code is not meant to be executable. This code is an outline of how to implement these types of models. For an example of a ConvLSTM that runs see my [collision anticipation](https://github.com/trevor-richardson/collision_anticipation) repo.

## Relevant Files

PyTorch implementation of ConvLSTM that initializes weights and defines forward method for inference
```
see pt_conv2dlstm_cell.py
```
<br/>

Tensorflow implementation of ConvLSTM static graph
```
see tf_conv2dlstm_cell.py
```
<br/>

PyTorch implementation of deep neural network containing a ConvLSTM Cell
```
see pytorch_model.py
```
<br/>

Tensorflow implementation of deep neural network containing a ConvLSTM Cell
```
see tensorflow_model.py
```

### Installing
Packages needed to run the code include:
* numpy
* python3
* PyToch
* Tensorflow
