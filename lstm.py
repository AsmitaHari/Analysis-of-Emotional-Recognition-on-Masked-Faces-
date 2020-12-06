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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from itertools import cycle
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score

input_size = 136
h1 = 120
output_dim = 8
num_layers = 2
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

def validation_metrics(model, dataset):
    '***Insert your code here'
    confusionMatrix = torch.zeros(8, 8)
    y_pred = []
    y_label = []
    y_test = []
    y_score = []
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    n_correct = 0
    n_samples = 0
    actuals = []
    probabilities  = []
    which_class = 7
    with torch.no_grad():

        for audio, labels in dataset:
            audio = audio
            labels = labels
            outputs = model(audio)
            prediction = outputs.argmax(dim=1, keepdim=True)
            actuals.extend(labels.view_as(prediction) == which_class)
            probabilities.extend(np.exp(outputs[:, which_class]))
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusionMatrix[t.long(), p.long()] += 1
            y_score.append(predicted.numpy())
            y_test.append(labels.numpy())
    print(y_score)
    print(y_test)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    cf = confusion_matrix(lbllist, predlist)
    sns.heatmap(cf, annot=True)
    plt.show()
    plot_roc([i.item() for i in actuals], [i.item() for i in probabilities],8)
    return acc, confusion_matrix


def plot_roc(y_test, y_score, N_classes):
    """
    compute ROC curve and ROC area for each class in each fold

    """
    #print('F1 score: %f' % f1_score(y_test, y_score, average='micro'))

    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for digit=%d class' % N_classes)
    plt.legend(loc="lower right")
    plt.show()
    # for i in range(N_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(N_classes):
    #     mean_tpr += interp1d(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= N_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(N_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

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

    n_total_steps = len(valid_loader)
    for epoch in range(num_epochs):
        for i, (audio, labels) in enumerate(valid_loader):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]

            # Forward pass\
            model.eval()
            outputs = model(audio)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    f'Epoch for validation [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    validation_metrics(model, test_loader)



if __name__ == '__main__':
    trainLoader = readCsv()
    lstm(trainLoader)
