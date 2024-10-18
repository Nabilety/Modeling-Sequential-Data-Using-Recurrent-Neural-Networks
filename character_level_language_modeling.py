import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions.categorical import Categorical
# ## Project two: character-level language modeling in PyTorch
# in character-level language modeling, the input is broken down into a sequence of characters
# that are fed into our network one character at a time. The network will process each new character
# in conjunction with the memory of the previously seen characters to predict the next one.
# The input is a text document, and our goal is to develop a model generating new text
# similar to the input document





# ### Preprocessing the dataset




## Reading and processing text
with open('1268-0.txt', 'r', encoding="utf8") as fp:
    text=fp.read()

# remove beginning and end portion
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')

text = text[start_indx:end_indx]
char_set = set(text) # represent unique characters observed in this text
print('Total Length:', len(text)) # sequence consisting of 1,112,350 characters
print('Unique Characters:', len(char_set)) #  80 unique characters

# Most NN libraries and RNN implementation can't deal with input data in strong format, which is why we have
# to convert the text into a numeric format. To do this we create a simple dictionary mappping each
# character to an integer, char2int. We also need a reverse mapping to convert the result of our model back to text
# Although the reverse can be done with a dictionary that associates integer keys with character values,
# using NumPy array and indexing the array to map indices to those unique character is more efficient.
chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32)

print('Text encoded shape: ', text_encoded.shape)

print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(char_array[text_encoded[15:21]]))



# text_encoded NumPy array contains the encoded values for all the characters in the text
# print out the mappings of the first five characters from this array
for ex in text_encoded[:5]:
    print('{} -> {}'.format(ex, char_array[ex]))


# To implement text generation in PyTorch, let's first clip the sequence length to 40 (good trade-off value).
# Meaning, input tensor, x consist of 40 tokens. In practice the sequence length impacts
# the quality of the generated text. Longer sequences can result in more meaningful sentences.
# For shorter sequences, however, the model might focus on capturing individual words correctly,
# while ignoring the context for the most part. Even though longer sequences are better for meaningful
# sentences, the RNN model will have problems capturing long-range dependencies. So in practice
# finding a sweet spot and good value for the sequence length is a hyperparameter optimization problem evaluated empirically

# First create chunks consisting of 41 characters each. Furthermore, get rid of the last chunk if it's shorter
# than 41 characters. The new chunked dataset will be named text_chunks containing sequences size 41 always.
# The 41 character chunks will then be used to construct the sequence x (that is, the input), as well as
# the sequence y (that is, the target), both of which will have 40 elements. I.e. sequence x will consist
# of elements with indices [0,1,...,39]. Furthermore, since sequence y will be shifted by one position
# with respect to x, its corresponding indices will be [1,2,...,40]. Then the result is transformed
# into  a Dataset object by applying a self-defined Dataset class:
seq_length = 40
chunk_size = seq_length + 1

text_chunks = [text_encoded[i:i+chunk_size]
               for i in range(len(text_encoded)-chunk_size+1)]

## inspection:
for seq in text_chunks[:1]:
    input_seq = seq[:seq_length]
    target = seq[seq_length]
    print(input_seq, ' -> ', target)
    print(repr(''.join(char_array[input_seq])),
          ' -> ', repr(''.join(char_array[target])))


class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


seq_dataset = TextDataset(torch.tensor(text_chunks))

for i, (seq, target) in enumerate(seq_dataset):
    print(' Input (x):', repr(''.join(char_array[seq])))
    print('Target (y):', repr(''.join(char_array[target])))
    print()
    if i == 1:
        break

device = torch.device("cuda:0")
# device = 'cpu'

batch_size = 64

torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ### Building a character-level RNN model
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(device), cell.to(device)

# Note the use of logits as outputs of the model so we can sample from model predictions in order to generate
# new text

# Specify model parameters and create RNN model
vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512

torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
model = model.to(device)
print(model)

# Create loss function and optimizer (Adam). For multiclass classification (we have vocab_size=80 classes)
# with a single logits output for  each target character, we use CrossEntropyloss as loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Now train the model for 10.000 epochs. Each epoch we wil use only one batch randomly chosen
# from the data loader, seq_dl. We will also display training loss for every 500 epochs
num_epochs = 10000

torch.manual_seed(1)
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    seq_batch = seq_batch.to(device)
    target_batch = target_batch.to(device)
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item() / seq_length
    if epoch % 500 == 0:
        print(f'Epoch {epoch} loss: {loss:.4f}')

# next we can evaluate the model to generate new text, starting with a given short string

# ### Evaluation phase: generating new text passages
# the RNN model from before returns logits of size 80 for each unique character. We convert these logits to probabilities
# via the softmax function that a particular character will be encountered as the next character. We simply predict
# the next character in the sequence as the element with maximum logit value, equivalent to selecting the character
# with the highest probability. However, instead of always selecting the character with the highest likelihood, we want
# to (randomly) sample from the outputs; otherwise, the model will always produce the same text.
# PyTorch already has a class for this, torch.distributions.vategorical.Categorical used to draw random samples from
# a categorical distribution.
# Let see how this works by generating random samples from three categories [0,1,2] with inputs [1,1,1]
torch.manual_seed(1)

logits = torch.tensor([[1.0, 1.0, 1.0]])

print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])

m = Categorical(logits=logits)
samples = m.sample((10,))

print(samples.numpy())

# So with a given logits, the categories have the same probabilities.
# Therefore, if we use a large sample size (num_samples → ∞), we would expect the number of occurrences of each
# category to reach approximately 1/3 of the sample size. If we change the logits to [1,1,3], then we would expect
# to observe more occurrences for category 2 (when a very large number of examplers are drawn from the distribution)

torch.manual_seed(1)

logits = torch.tensor([[1.0, 1.0, 3.0]])

print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])

m = Categorical(logits=logits)
samples = m.sample((10,))

print(samples.numpy())

# Using Categorical, we can generate examples based on the logits computed by our model

# Define function with a short starting string, starting_str as input
# generating a new string, generated_str as output (generated_str is initially set to the input string)
# starting_str is encoded to a sequence of integers, encoded_input. encoded_input is passed to the RNN model one character
# at a time to update the hidden states. The last character of encoded_input is passed to the model to generate a new
# character. Note that the output of the RNN model represents the logits (here, a vector of size 80, total number of possible characters)
# for the next character after observing the input sequence by the model. Here we only use a logits output, which is
# passed to the Categorical class to generate a new sample. This new sample is converted to a character, which is then
# appended to the end of the generated string, generated_text, increasing its length by 1. Rinse and repeat until the
# length of the generated string reaches the desired value. The process of consuming the generated sequence as input
# for generating new elements is called autoregression
def sample(model, starting_str,
           len_generated_text=500,
           scale_factor=1.0):
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden = hidden.to('cpu')
    cell = cell.to('cpu')
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

    return generated_str


torch.manual_seed(1)
model.to('cpu')
print(sample(model, starting_str='The island'))

# * **Predictability vs. randomness**
# As shown the model generates mostly correct words and in some cases the sentences are partially meaningful.
# The training parameters can be further tuned, such as the length of input sequences for training and model architecture

# To control predictability of the generated samples (that is, generating text following hte learned patterns from the
# training text versus adding more randomness), the logits computed by the RNN model can be scaled before being passed
# to Categorical for sampling. Scaling factor 𝛼 can be interpreted as an analog to the temperature in physics.
# Higher temperature results in more entropy or randomness versus more predictable behavior at lower temperature.
# BY scaling the logits with 𝛼 < 1, the probabilities computed by the softmax function become more uniform:

logits = torch.tensor([[1.0, 1.0, 3.0]])

print('Probabilities before scaling:        ', nn.functional.softmax(logits, dim=1).numpy()[0])

print('Probabilities after scaling with 0.5:', nn.functional.softmax(0.5*logits, dim=1).numpy()[0])

print('Probabilities after scaling with 0.1:', nn.functional.softmax(0.1*logits, dim=1).numpy()[0])


# As you can see, scaling the logits by 𝛼 results in near-uniform probabilities [0.31, 0.31, 0.38]. Now,
# we can compare the generated text with 𝛼 = 2.0 and 𝛼 = 0.5, as shown in the following points:

# 𝛼 = 2.0 → more predictable:
torch.manual_seed(1)
print(sample(model, starting_str='The island',
             scale_factor=2.0))



# 𝛼 = 0.5 → more randomness:
torch.manual_seed(1)
print(sample(model, starting_str='The island',
             scale_factor=0.5))


