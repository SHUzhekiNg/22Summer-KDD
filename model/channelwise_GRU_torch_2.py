import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class GRUModel(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, embedding_dim, hidden_units):
        super(GRUModel, self).__init__()
        self.embedding1 = nn.Embedding(vocab_size1, embedding_dim)
        self.embedding2 = nn.Embedding(vocab_size2, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_units, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, hidden_units, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(2 * hidden_units, vocab_size1)

    def forward(self, input1, input2):
        embedded_input1 = self.embedding1(input1)
        embedded_input2 = self.embedding2(input2)
        gru_output1, _ = self.gru1(embedded_input1)
        gru_output2, _ = self.gru2(embedded_input2)
        dropout_output1 = self.dropout(gru_output1[:, -1, :])
        dropout_output2 = self.dropout(gru_output2[:, -1, :])
        concatenated_output = torch.cat((dropout_output1, dropout_output2), dim=1)
        output = self.output_layer(concatenated_output)
        return output


class Metesre():
    def __init__(self, embedding_dim, hidden_units, max_len):
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_len = max_len
        item_ids = pd.read_parquet('../preprocessing/categories/unique.prev_items.parquet')
        self.items = item_ids['prev_items']
        self.sessions_gdf = pd.read_parquet("../processed_nvt/part_0.parquet")
        self.test_gdf = pd.read_parquet("../processed_nvt/test_0.parquet")
        self.preprocessing()
        self.buildModel()

    def preprocessing(self):
        X1 = self.sessions_gdf['prev_items-list'].tolist()
        X2 = self.sessions_gdf['title-list'].tolist()

        X1 = np.array(X1, dtype='object')

        self.vocab_size1 = max(item for sublist in X1 for item in sublist) + 1
        self.vocab_size2 = max(item for sublist in X2 for item in sublist) + 1

        print("Vocab Sizes: \n", self.vocab_size1, self.vocab_size2)

        X1_p = []
        X2_p = []
        y_p = []

        for i in range(len(X1)):
            X1_p.append(X1[i][:-1])
            X2_p.append(X2[i][:-1])
            y_p.append(X1[i][-1])

        X1 = X1_p
        X2 = X2_p
        y = np.array(y_p)

        self.max_len = 10

        X1 = pad_sequence([torch.tensor(seq) for seq in X1], batch_first=True, padding_value=0)
        X2 = pad_sequence([torch.tensor(seq) for seq in X2], batch_first=True, padding_value=0)

        self.X1_train, self.X1_test, self.X2_train, self.X2_test, self.y_train, self.y_test = train_test_split(
            X1, X2, y, test_size=0.005, random_state=42, shuffle=True)

    def buildModel(self):
        self.model = GRUModel(self.vocab_size1, self.vocab_size2, self.embedding_dim, self.hidden_units)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self, epoch=10, batch_size=32):
        self.model.train()
        train_dataset = TensorDataset(self.X1_train, self.X2_train, torch.tensor(self.y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epoch):
            running_loss = 0.0
            for inputs1, inputs2, labels in tqdm(train_loader):
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs1, inputs2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    def test(self):
        self.model.eval()
        test_dataset = TensorDataset(self.X1_test, self.X2_test, torch.tensor(self.y_test))
        test_loader = DataLoader(test_dataset, batch_size=1)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs1, inputs2, labels in test_loader:
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs1, inputs2)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total}%")
