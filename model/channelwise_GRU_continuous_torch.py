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
    def __init__(self, vocab_size1, vocab_size2, vocab_size3, vocab_size4, vocab_size5, vocab_size6, embedding_dim, hidden_units):
        super(GRUModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding1 = nn.Embedding(vocab_size1, embedding_dim)
        self.embedding2 = nn.Embedding(vocab_size2, embedding_dim)
        self.embedding3 = nn.Embedding(vocab_size3, embedding_dim)
        self.embedding4 = nn.Embedding(vocab_size4, embedding_dim)
        self.embedding5 = nn.Embedding(vocab_size5, embedding_dim)
        self.embedding6 = nn.Embedding(vocab_size6, embedding_dim)

        self.gru = nn.GRU(self.embedding_dim * 6 + 2, hidden_units, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_units, vocab_size1)

    def forward(self, input1, input2, input3, input4, input5, input6, inputs7, inputs8):
        embedded_input1 = self.embedding1(input1)
        embedded_input2 = self.embedding2(input2)
        embedded_input3 = self.embedding3(input3)
        embedded_input4 = self.embedding4(input4)
        embedded_input5 = self.embedding5(input5)
        embedded_input6 = self.embedding6(input6)

        inputs7_cont = inputs7.unsqueeze(2)
        inputs8_cont = inputs8.unsqueeze(2)
        emb_cont7 = nn.Linear(1, self.embedding_dim).to(inputs7.dtype).to(inputs7.device)(inputs7_cont)
        emb_cont8 = nn.Linear(1, self.embedding_dim).to(inputs8.dtype).to(inputs8.device)(inputs8_cont)
        
        concatenated_output = torch.cat((embedded_input1, embedded_input2, embedded_input3, embedded_input4, embedded_input5, embedded_input6, emb_cont7, emb_cont8), dim=1)

        gru_output, _ = self.gru(concatenated_output)
        gru_output = self.dropout(gru_output)
        output = self.output_layer(gru_output)

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
        print("start preprocessing.")
        X1 = self.sessions_gdf['prev_items-list'].tolist()
        X2 = self.sessions_gdf['title-list'].tolist()
        X3 = self.sessions_gdf['brand-list'].tolist()
        X4 = self.sessions_gdf['size-list'].tolist()
        X5 = self.sessions_gdf['model-list'].tolist()
        X6 = self.sessions_gdf['color-list'].tolist()
        X7 = self.sessions_gdf['price_log_norm-list'].tolist()
        X8 = self.sessions_gdf['relative_price_to_avg_categ_id-list'].tolist()

        X1 = np.array(X1, dtype='object')

        self.vocab_size1 = max(item for sublist in X1 for item in sublist) + 1
        self.vocab_size2 = max(item for sublist in X2 for item in sublist) + 1
        self.vocab_size3 = max(item for sublist in X3 for item in sublist) + 1
        self.vocab_size4 = max(item for sublist in X4 for item in sublist) + 1
        self.vocab_size5 = max(item for sublist in X5 for item in sublist) + 1
        self.vocab_size6 = max(item for sublist in X6 for item in sublist) + 1

        print("Vocab Sizes: \n", self.vocab_size1, self.vocab_size2, self.vocab_size3, self.vocab_size4, self.vocab_size5, self.vocab_size6)

        X1_p = []
        X2_p = []
        X3_p = []
        X4_p = []
        X5_p = []
        X6_p = []
        X7_p = []
        X8_p = []
        y_p = []

        for i in range(len(X1)):
            X1_p.append(X1[i][:-1])
            X2_p.append(X2[i][:-1])
            X3_p.append(X3[i][:-1])
            X4_p.append(X4[i][:-1])
            X5_p.append(X5[i][:-1])
            X6_p.append(X6[i][:-1])
            X7_p.append(X7[i][:-1])
            X8_p.append(X8[i][:-1])
            y_p.append(X1[i][-1])

        X1 = X1_p
        X2 = X2_p
        X3 = X3_p
        X4 = X4_p
        X5 = X5_p
        X6 = X6_p
        X7 = X7_p
        X8 = X8_p
        y = np.array(y_p)

        # X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        # X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
        # X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
        # X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
        # X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
        # X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')

        X1 = pad_sequence([torch.tensor(seq) for seq in X1], batch_first=True, padding_value=0)
        X2 = pad_sequence([torch.tensor(seq) for seq in X2], batch_first=True, padding_value=0)
        X3 = pad_sequence([torch.tensor(seq) for seq in X3], batch_first=True, padding_value=0)
        X4 = pad_sequence([torch.tensor(seq) for seq in X4], batch_first=True, padding_value=0)
        X5 = pad_sequence([torch.tensor(seq) for seq in X5], batch_first=True, padding_value=0)
        X6 = pad_sequence([torch.tensor(seq) for seq in X6], batch_first=True, padding_value=0)
        X7 = pad_sequence([torch.tensor(seq) for seq in X7], batch_first=True, padding_value=0)
        X8 = pad_sequence([torch.tensor(seq) for seq in X8], batch_first=True, padding_value=0)

        self.X1_train, self.X1_test, self.X2_train, self.X2_test, self.X3_train, self.X3_test,\
        self.X4_train, self.X4_test, self.X5_train, self.X5_test, self.X6_train, self.X6_test,\
        self.X7_train, self.X7_test, self.X8_train, self.X8_test, self.y_train, self.y_test = train_test_split(
            X1, X2, X3, X4, X5, X6, X7, X8, y, test_size=0.005, random_state=42, shuffle=True)
        print("preprocessing done.")

    def buildModel(self):
        self.model = GRUModel(self.vocab_size1, self.vocab_size2, self.vocab_size3, self.vocab_size4, self.vocab_size5, self.vocab_size6, self.embedding_dim, self.hidden_units)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self, epoch=10, batch_size=32):
        self.model.train()
        train_dataset = TensorDataset(self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train, self.X6_train, self.X7_train, self.X8_train, torch.tensor(self.y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        best_loss = 10000000
        for epoch in range(epoch):
            running_loss = 0.0
            for inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, labels in tqdm(train_loader):
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)
                inputs3 = inputs3.to(self.device)
                inputs4 = inputs4.to(self.device)
                inputs5 = inputs5.to(self.device)
                inputs6 = inputs6.to(self.device)
                inputs7 = inputs7.to(self.device)
                inputs8 = inputs8.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader)}")
            if best_loss > running_loss:
                best_loss = running_loss
                print("saving to model.pth")
                torch.save(self.model.state_dict(),'model.pth')

    def test(self):
        self.model.eval()
        test_dataset = TensorDataset(self.X1_test, self.X2_test, self.X3_test,
                                     self.X4_test, self.X5_test, self.X6_test,
                                     self.X7_test, self.X8_test, torch.tensor(self.y_test))
        test_loader = DataLoader(test_dataset, batch_size=1)

        correct = 0
        total = 0
        reciprocal_ranks = []

        with torch.no_grad():
            for inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, labels in tqdm(test_loader):
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)
                inputs3 = inputs3.to(self.device)
                inputs4 = inputs4.to(self.device)
                inputs5 = inputs5.to(self.device)
                inputs6 = inputs6.to(self.device)
                inputs7 = inputs7.to(self.device)
                inputs8 = inputs8.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 将计算转移到CUDA设备上
                scores = outputs.data.squeeze().cpu().numpy()
                ranks = (-scores).argsort()
                rank_of_correct = np.where(ranks == labels.item())[0][0]
                reciprocal_rank = 1.0 / (rank_of_correct + 1)
                reciprocal_ranks.append(reciprocal_rank)

        accuracy = 100 * correct / total
        mrr = np.mean(reciprocal_ranks)

        print(f"Test Accuracy: {accuracy}%")
        print(f"MRR: {mrr}")
