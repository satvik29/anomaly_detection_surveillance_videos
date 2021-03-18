import torch
import torch.nn as nn
import torch.nn.init as torch_init
import tensorflow as tf
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

# class Model(nn.Module):
#     def __init__(self, n_features):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(n_features, 512)
#         self.fc2 = nn.Linear(512, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.dropout = nn.Dropout(0.6)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.apply(weight_init)
#
#     def forward(self, inputs):
#         x = self.relu(self.fc1(inputs))
#         x = self.dropout(x)
#         hidden = x
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.sigmoid(self.fc3(x))
#         x = self.dropout(x)
#         return x

class Model(nn.Module):
     def __init__(self, n_features):
         super(Model, self).__init__()
         self.embedding = nn.Embedding(n_features, 512)
         #self.lstm1 = nn.LSTM(512, 512)
         #self.lstm2 = nn.LSTM(512, 32)
         #self.lstm3 = nn.LSTM(32, 1)
         self.lstm1 = nn.LSTM(512, 512,bidirectional=True)
         self.lstm2 = nn.LSTM(1024, 32,bidirectional=True)
         self.lstm3 = nn.LSTM(64, 1,bidirectional=True)
         self.fc = nn.Linear(2, 1)
         #self.fc = nn.Linear(1,1)
         self.dropout = nn.Dropout(0.6)
         self.relu = nn.ReLU()
         self.sigmoid = nn.Sigmoid()
         #self.apply(weight_init) ---> not sure if and how we need this

     def forward(self, inputs):
         inputs = inputs.long()
         inputs = torch.squeeze(inputs)
         x = self.embedding(inputs)
         #LSTM layers here with dropout after first 2 layers.
         # Dropout is being individually applied here instead of in the init() since
         # it's applied to  only the last layer with the LSTM initializer
         x,(hidden_layer, cell_memory) = self.lstm1(x)
         #x = self.relu(self.lstm1(x))
         x = self.relu(x)
         x = self.dropout(x)
         x,(hidden_layer, cell_memory) = self.lstm2(x)
         x = self.dropout(x)
         x,(hidden_layer, cell_memory) = self.lstm3(x)
         x = self.sigmoid(x)
         output = torch.cat((hidden_layer[-2, :, :], hidden_layer[-1, :, :]), dim=1)
         x = self.fc(output)
         x = x.unsqueeze(0)
         #print(x)
         return x
