from pt_conv2dlstm_cell import StatefulConv2dLSTMCell
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Define the neural network and use the custom built Conv2dLSTM'''
class ExampleConvLSTM2D(nn.Module):
    def __init__(self, input_shp, no_filters, kernel_size, strides, output_shp, padding=0, dropout_rte=0):
        super(Custom_Spatial_Temporal_Anticipation_NN, self).__init__()
        print("Running Custom Built Stateful Conv2dLSTM")

        self.convlstm_0 = StatefulConv2dLSTMCell(input_shp, no_filters[0], kernel_size, strides, pad=padding)

        self.convlstm_1 = StatefulConv2dLSTMCell(self.convlstm_0.output_shape, no_filters[1], kernel_size, strides, pad=padding)

        self.convlstm_2 = StatefulConv2dLSTMCell(self.convlstm_1.output_shape, no_filters[2], kernel_size, strides, pad=padding)

        flat = self.convlstm_2.output_shape[0] * self.convlstm_2.output_shape[1] * self.convlstm_2.output_shape[2]

        self.dropout = nn.Dropout(dropout_rte)
        self.fcn1 = nn.Linear(flat, output_shp)

    def forward(self, x, states):

        hx_0, cx_0 = self.convlstm_0(x, [states[0][0] ,states[0][1]])
        hx_1, cx_1 = self.convlstm_1(hx_0, (states[1][0] ,states[1][1]))
        hx_2, cx_2 = self.convlstm_2(hx_1, (states[2][0] ,states[2][1]))
        dropped = self.dropout(hx_2.view(hx_2.size(0), -1)) #use dropout on flattened output of convlstm cell
        y = F.sigmoid(self.fcn1(dropped))
        return y, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]]
