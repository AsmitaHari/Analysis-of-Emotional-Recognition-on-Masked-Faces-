import os
import pandas as pd
import numpy as np
import torch
import glob
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import csv

input_size = 136
h1 = 120
output_dim = 8
num_layers = 3
num_epochs= 100
class SpeechLoader(Dataset):
    def __init__(self, dataset_file):
        self.label = list()
        self.dataset = list()
        try:
            with open(dataset_file, 'r') as f:
                csv_reader = csv.reader(f)
                for dat in csv_reader:
                    self.label.append(torch.tensor([int(dat[0])]))
                    np_array = np.array(dat[1:], dtype=np.float32).reshape(input_size, -1)
                    self.dataset.append(torch.from_numpy(np_array).permute(1, 0).reshape(-1, 1, input_size))
        except FileNotFoundError:
            print('generate features for [' + dataset_file + ']')
            exit(1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]

    def to(self, device):
        for i in range(len(self.dataset)):
            self.label[i] = self.label[i].to(device=device)
            self.dataset[i] = self.dataset[i].to(device=device)
        return self


def lstm_style_batching(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    data = torch.cat(data, dim=1)
    label = torch.cat(label, dim=0)
    return data, label

def readCsv():
    list = []

    finalList = np.array(list)
    count = 0
    with open("featureLabel.csv", 'w') as f:
        writer = csv.writer(f)
        dataList = []
        for filepath in glob.glob('C:/Users/asmit/PycharmProjects/pytorch/739-proj/outFolder/**/**/landmark/*.csv'):
            dirName = os.path.dirname(filepath)
            length = len(glob.glob(dirName + "/*.csv"))

            if (length == 20):
                df = pd.read_csv(filepath)
                df = df.drop(df.columns[0], axis=1)
                dataFrame = df.to_numpy()


                if (count==0):
                    x = np.array([])
                resized = dataFrame.reshape(68,1,2)
                x = resized if x.size == 0 else np.vstack((x, resized))
                count += 1

                if (count==20):
                    count = 0
                    flattenArray = np.ndarray.flatten(x)
                    fileName = os.path.basename(filepath)
                    dir = fileName.split("_")
                    dirPath = os.path.join("C:/RIT-Stuff/Topics in System/Project/Emotion_labels/Emotion/", dir[0], dir[1])
                    text = 0
                    for filepath in glob.glob(dirPath + "/*.txt"):
                        file1 = open(filepath, "r+")
                        splitVal = file1.read().split(".")
                        text = int(splitVal[0])
                    labelledList = np.append(text,flattenArray)
                    list.append(labelledList)
        print(len(list))
        writer.writerows(list)
        print(f'Done with features')

    finalList = torch.tensor(list)
    dataSet = TensorDataset(finalList)
    trainLoader = DataLoader(dataset=dataSet, shuffle=True)
    dataFrame = pd.read_csv('featureLabel.csv', sep=",")
    train_size = 0.8
    validate_size = 0.1
    train, valid, test = np.split(dataFrame.sample(frac=1), [int(train_size * len(dataFrame)),
                                                             int((validate_size + train_size) * len(dataFrame))])
    train.to_csv('mfcc_train.csv',index=False)
    test.to_csv('mfcc_test.csv', index=False)
    valid.to_csv('mfcc_valid.csv',index=False)

    return trainLoader
def lstm(trainLoader) :
    train_loader = DataLoader(SpeechLoader("mfcc_train.csv"), batch_size=16, collate_fn=lstm_style_batching,
                              shuffle=True)
    test_loader = DataLoader(SpeechLoader("mfcc_test.csv"), batch_size=16, collate_fn=lstm_style_batching, shuffle=True)
    valid_loader = DataLoader(SpeechLoader("mfcc_valid.csv"), batch_size=16, collate_fn=lstm_style_batching,
                              shuffle=True)
    class LSTMSpeechEmo(nn.Module):
        def __init__(self, input_dim, hidden_dim, target_size, num_lstm_layers):
            '***Insert your code here'
            super(LSTMSpeechEmo, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_lstm_layers = num_lstm_layers
            self.output_dim = target_size
            # self.act= nn.ReLU()

            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_lstm_layers)
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)
            self.softmax = nn.LogSoftmax()
            self.dropout_layer = nn.Dropout(p=0.2)

        def forward(self, x):
            '***Insert your code here'
            #         h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim)
            #         c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim)

            out, (ht, ct) = self.lstm(x, None)
            out = out[-1, :]
            output = self.dropout_layer(ht[-1])
            out = self.fc(out)

            return out

    learning_rate = 0.001

    loss_fn = nn.CrossEntropyLoss()
    model = LSTMSpeechEmo(input_size, h1, output_dim, num_layers)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    #####################
    # Train model
    #####################

    '***Insert your code here'

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        totalLoss = 0.0
        for i, (audio, labels) in enumerate(train_loader):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]

            # Forward pass\
            model.train()
            outputs = model(audio)
            loss = loss_fn(outputs, labels)
            totalLoss = totalLoss + loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')



if __name__ == '__main__':
    trainLoader = readCsv()
    lstm(trainLoader)