import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
print(torch.cuda.is_available())
print(torch.version.cuda)
if torch.cuda.is_available():
    print("Cuda is Availabe")
else:
    print("Cuda Can't be found")

# # Implementing RNNs for sequence modeling in PyTorch
#
# ## Project one: predicting the sentiment of IMDb movie reviews
# multilayer RNN for sentiment analysis using a many-to-one architecture
#device = torch.device("cpu")
# ### Preparing the movie review data
#
# Step 1: load and create the datasets
# each set has 25000 samples, and each sample consist of two elements;
# the sentiment label representing the target label we want to predict (neg refer to negative and pos for positive)
# and the movie review text (input features)
train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')



# Preprocess / clean review dataset:
# Step 2: Split training dataset into separate training and validation partitions
# Original dataset contains 25000 examples. 20000 examples are randomly chose for training
# while the 5000 rest for validation
torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])

# Step 3: Identify the unique words in the training dataset - using Counter class from collections package
# we will instantiate new Counter object (token_counts) collecting unique word frequencies. Note this
# particular application  (in contrast to bag-of-words model) we only interested in set of unique words
# and won't require the word counts, which are created as a side product. To split text into words (or tokens)
# we will reuse the tokenizer function, which also removes HTML markups as well as punctuation and non-letter characters
def tokenizer(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized

token_counts = Counter()

for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
print('Vocab-size:', len(token_counts)) # number of distinct words

## Step 3: encoding each unique token into integers
# Map each unique word to a unique integer and encode the review text into encoded integers
# (an index of each unique word). Can be done simply with Python dict with keys as unique tokens (words)
# and values associated with each key is a unique integer. However torchtext package already provides a class
# Vocab used to create such a mapping and encode the entire dataset. We create a vocab object by passing
# the ordered dictionary mapping tokens to their corresponding occurence frequencies
# (the ordered dictionary is the sorted token_counts). Secondly we prepend two special tokens to the vocabulary
# - namely, padding and the unknown token
sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)

vocab = vocab(ordered_dict)

vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

print([vocab[token] for token in ['this', 'is', 'an', 'example']]) # Note this is how you use the vocab object, and as shown the word 'this' has index 11

# note there might be some tokens in the validation or testing data that are not present in the training data
# and are thus not included in the mapping. If we have q tokens (that is, size of token_counts passed to Vocab, which is 69023)
# then all tokens that haven't been seen before, and are thus not included in token_counts, will be assigned the integer 1
# (a placeholder for the unknown token). In other words, index 1 is reserved for unknown words. Another reserved value
# is the integer 0, which serves as a placeholder, a so-called padding token for adjusting the sequence length.

## Step 3-A: Define the functions for transformation
# define text_pipeline function to transform each text in the dataset accordingly and the label_pipeline
# function to convert label to 1 or 0:
device = torch.device("cuda:0")
# device = 'cpu'
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 2 else 0. # note for torchtext.datasets the return value are 2 and 1, but in the real dataset it's 'pos' and 'neg'


## Step 3-B: Wrap the encode and transformation function.
# We will generate batches of samples using DataLoader and pass the data processing pipelines declared
# previously to the argument collate_fn. We will wrap the text encoding and label transformation
# function into the collate_batch function:
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text),
                                      dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    # ensuring all sequences in mini-batch have same length to store them efficiently in a tensor,
    # by padding consecutive elements that are to be combined into a batch with placeholder values
    # (0s) so all sequences within a batch wil have the same shape
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True)
    return padded_text_list.to(device), label_list.to(device), lengths.to(device)


## Step 4: batching the datasets
# Divide the dataset into mini-batches as input to the model
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)


## Divide all three datasets into data loaders with batch size 32

batch_size = 32

train_dl = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_batch)

# Now that the data is in a suitable format for an RNN model, we are going to implement in the following subsections
# However we will first discuss feature embedding, which is optional but highly recommended preporcessing step
# used to reduce the dimensionality of the word vectors

# ### Embedding layers for sentence encoding
#
#
#  * `input_dim`: number of words, i.e. maximum integer index + 1.
#  * `output_dim`:
#  * `input_length`: the length of (padded) sequence
#     * for example, `'This is an example' -> [0, 0, 0, 0, 0, 0, 3, 1, 8, 9]`
#     => input_lenght is 10
#
#
#
#  * When calling the layer, takes integr values as input,
#  the embedding layer convert each interger into float vector of size `[output_dim]`
#    * If input shape is `[BATCH_SIZE]`, output shape will be `[BATCH_SIZE, output_dim]`
#    * If input shape is `[BATCH_SIZE, 10]`, output shape will be `[BATCH_SIZE, 10, output_dim]`


embedding = nn.Embedding(num_embeddings=10, # unique integer values the model receives as input (i.e. n+2, set to 10 here)
                         embedding_dim=3, # size of embedding features
                         padding_idx=0) # token index for padding, here 0 as we defined above

# a batch of 2 samples of 4 indices each
text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
print(embedding(text_encoded_input))

# therefore the embedding matrix in this case has size 10 x 6 = 10 x 4 + 2 = 10 x n + 2
# in our example the original sequence of the second sample is 3, and we padded it with 1 more element 0.
# The embedding output of the padded element is [0, 0, 0]

# ### Building an RNN model
#
# * **RNN layers:**
#   * `nn.RNN(input_size, hidden_size, num_layers=1)`
#   * `nn.LSTM(..)`
#   * `nn.GRU(..)`
#   * `nn.RNN(input_size, hidden_size, num_layers=1, bidirectional=True)`
#
#



## An example of building a RNN model
## with simple RNN layer, with two reccurent layers and finally add a non-reccurent fully connected layer as output layer

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # recurrent layer
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          num_layers=2,
                          batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # non-recurrent layer

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1, :, :] # we use the final hidden state
                               # from the last hiden layer as
                               # the input to the fully connected layer
        out = self.fc(out)
        return out


model = RNN(64, 32)

print(model)

model(torch.randn(5, 3, 64))

# ### Building an RNN model for the sentiment analysis task
# Since we have very long sequences, we are going to use an LSTM layer to account for long-range effects
# we will create an RNN model for sentiment analysis, starting with an embedding layer producing word embeddings
# of feature size 10 (embed_dim=20). Then a recurrent layer of type LSTM will be added. Finally we will add a
# fully connected layer as a hidden layer and another fully connected layer as the output layer returning a s
# single class-membership probability value via the logistic sigmoid activation as prediction
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim,
                                      padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                           batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64

torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
model = model.to(device)
print(model)

# Create loss function and optimizer (Adam). For binary classification with a single class-membership probability output
# we use the binary cross-entropy loss (BCELoss) as loss function.
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define train function to train the model on given dataset for one epoch and return classification accuracy and loss
def train(dataloader):
    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
        total_count += label_batch.size(0)
    return total_acc/total_count, total_loss/total_count

# Define evaluate function to measure model's performance on given (test) dataset
def evaluate(dataloader):
    model.eval()
    total_acc, total_loss, total_count = 0, 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
            total_count += label_batch.size(0)
    return total_acc/total_count, total_loss/total_count


# Now we train the model for 10 epochs and display the training and validation performance
num_epochs = 10

torch.manual_seed(1)

for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f}')

acc_test, _ = evaluate(test_dl)
print(f'test_accuracy: {acc_test:.4f}')
# It shows 0.8477 circa 85% accuracy
# (note this result is not the best when compared to the state-of-the-art methods used on the IMDb dataset.
# The goal was simply to show how an RNN works in PyTorch)

# ## Optional exercise:
#
# ### Uni-directional SimpleRNN with full-length sequences

# ## Project two:
# many-to-many RNN for an application of language modeling