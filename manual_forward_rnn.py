import torch
import torch.nn as nn


# Use torch.nn module create a recurrent layer from RNN and perform a forward pass on an input sequence
# of length 3 to compute the output. Manually compute the forward pass and compare results with those of RNN

torch.manual_seed(1)

rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)

w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

print('W_xh shape:', w_xh.shape)
print('W_hh shape:', w_hh.shape)
print('b_xh shape:', b_xh.shape)
print('b_hh shape:', b_hh.shape)

# Input shape of this layer is (batch_size, seuqence_length, 5)
# First dimension is batch dimension (set o True)
# Second dimension corresponds to the sequence
# Last dimension corresponds to the features
# Note we will output a sequence, which for an input sequence of length 3
# will result in output sequence 〈o^(0), o^(1), o^(2)〉
# Also RNN uses one layer by defuat, and u can set num_layers to stack multiple RNN layers together to form stacked RNN


# Call forward pass on the rnn_layer and manually compute outputs at each time step and compare them
x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()

## Output of the simple RNN:
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))

# manually compute the output:
out_man = []
for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))
    print(f'Time step {t} =>')
    print('   Input           :', xt.numpy())

    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
    print('   Hidden          :', ht.detach().numpy())

    if t > 0:
        prev_h = out_man[t - 1]
    else:
        prev_h = torch.zeros((ht.shape))

    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh
    ot = torch.tanh(ot)
    out_man.append(ot)
    print('   Output (manual) :', ot.detach().numpy())
    print('   RNN output      :', output[:, t].detach().numpy())
    print()

# in out manual forward computation, we used the hyperbolic tangent (tanh) activation function since it
# is also used in RNN (default activation). As we see the printed results the outputs from the manual
# forward computations excatly match the output of the RNN layer at each time step