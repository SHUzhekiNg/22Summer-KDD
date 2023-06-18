import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# corresponding to model
import torch
import torch.nn as nn


def preprocessing(max_len=10):
    item_ids = pd.read_parquet('../preprocessing/categories/unique.prev_items.parquet')
    items = item_ids['prev_items']
    sessions_gdf = pd.read_parquet("../processed_nvt/part_0.parquet")
    test_gdf = pd.read_parquet("../processed_nvt/test_0.parquet")

    X1 = sessions_gdf['prev_items-list'].tolist()
    X2 = sessions_gdf['title-list'].tolist()
    X3 = sessions_gdf['brand-list'].tolist()
    X4 = sessions_gdf['size-list'].tolist()
    X5 = sessions_gdf['model-list'].tolist()
    X6 = sessions_gdf['color-list'].tolist()

    # handles variable length session sequences
    X1 = np.array(X1, dtype='object')
    vocab_size = []
    # find vocab sizes
    vocab_size.append(max(item for sublist in X1 for item in sublist) + 1)
    vocab_size.append(max(item for sublist in X2 for item in sublist) + 1)
    vocab_size.append(max(item for sublist in X3 for item in sublist) + 1)
    vocab_size.append(max(item for sublist in X4 for item in sublist) + 1)
    vocab_size.append(max(item for sublist in X5 for item in sublist) + 1)
    vocab_size.append(max(item for sublist in X6 for item in sublist) + 1)

    print("Vocab Sizes: \n", vocab_size)

    # extract next item from the X1: prev_items_list, also remove last items attributes
    X1_p = []
    X2_p = []
    X3_p = []
    X4_p = []
    X5_p = []
    X6_p = []

    y_p = []

    for i in range(len(X1)):
        X1_p.append(X1[i][:-1])
        X2_p.append(X2[i][:-1])
        X3_p.append(X3[i][:-1])
        X4_p.append(X4[i][:-1])
        X5_p.append(X5[i][:-1])
        X6_p.append(X6[i][:-1])
        y_p.append(X1[i][-1])

    X1 = X1_p
    X2 = X2_p
    X3 = X3_p
    X4 = X4_p
    X5 = X5_p
    X6 = X6_p
    y = y_p
    y = np.array(y)

    # padding: pre for X1 and post for all others
    X1 = pad_sequences(X1, maxlen=max_len, padding='pre')
    X2 = pad_sequences(X2, maxlen=max_len, padding='pre')
    X3 = pad_sequences(X3, maxlen=max_len, padding='pre')
    X4 = pad_sequences(X4, maxlen=max_len, padding='pre')
    X5 = pad_sequences(X5, maxlen=max_len, padding='pre')
    X6 = pad_sequences(X6, maxlen=max_len, padding='pre')

    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
        X4_test, X5_train, X5_test, X6_train, X6_test, y_train, y_test = train_test_split(
        X1, X2, X3, X4, X5, X6, y, test_size=0.005, random_state=42, shuffle=True)

    return [X1_train,X2_train,X3_train,X4_train,X5_train,X6_train], y_train, \
        [X1_test,X2_test,X3_test,X4_test,X5_test,X6_test], y_test,\
        vocab_size
class Model(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, hidden_units, max_len):
        super(Model, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_len = max_len

        # Define the embedding layers
        self.embedding_layer1 = nn.Embedding(self.vocab_sizes[0], self.embedding_dim)
        self.embedding_layer2 = nn.Embedding(self.vocab_sizes[1], self.embedding_dim)
        self.embedding_layer3 = nn.Embedding(self.vocab_sizes[2], self.embedding_dim)
        self.embedding_layer4 = nn.Embedding(self.vocab_sizes[3], self.embedding_dim)
        self.embedding_layer5 = nn.Embedding(self.vocab_sizes[4], self.embedding_dim)
        self.embedding_layer6 = nn.Embedding(self.vocab_sizes[5], self.embedding_dim)

        # Define the GRU layers
        self.gru_layer1 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)
        self.gru_layer2 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)
        self.gru_layer3 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)
        self.gru_layer4 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)
        self.gru_layer5 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)
        self.gru_layer6 = nn.GRU(self.embedding_dim, self.hidden_units, batch_first=True)

        # Define the dropout layer
        self.dropout_layer = nn.Dropout(0.3)

        # Define the output layer
        self.output_layer = nn.Linear(self.hidden_units * 6, self.vocab_sizes[0])

    def forward(self, input1, input2, input3, input4, input5, input6):
        # Apply the embedding layers
        embedded_input1 = self.embedding_layer1(input1)
        embedded_input2 = self.embedding_layer2(input2)
        embedded_input3 = self.embedding_layer3(input3)
        embedded_input4 = self.embedding_layer4(input4)
        embedded_input5 = self.embedding_layer5(input5)
        embedded_input6 = self.embedding_layer6(input6)

        # Apply the GRU layers with dropout to the embedded inputs
        gru_output1, _ = self.gru_layer1(embedded_input1)
        dropout_output1 = self.dropout_layer(gru_output1[:, -1, :])

        gru_output2, _ = self.gru_layer2(embedded_input2)
        dropout_output2 = self.dropout_layer(gru_output2[:, -1, :])

        gru_output3, _ = self.gru_layer3(embedded_input3)
        dropout_output3 = self.dropout_layer(gru_output3[:, -1, :])

        gru_output4, _ = self.gru_layer4(embedded_input4)
        dropout_output4 = self.dropout_layer(gru_output4[:, -1, :])

        gru_output5, _ = self.gru_layer5(embedded_input5)
        dropout_output5 = self.dropout_layer(gru_output5[:, -1, :])

        gru_output6, _ = self.gru_layer6(embedded_input6)
        dropout_output6 = self.dropout_layer(gru_output6[:, -1, :])

        # Concatenate the GRU outputs
        concatenated_output = torch.cat(
            (dropout_output1, dropout_output2, dropout_output3, dropout_output4, dropout_output5, dropout_output6),
            dim=1)



    def _mean_reciprocal_rank(self, recommendations, ground_truth):
        """
        Calculate the Mean Reciprocal Rank (MRR) of a recommendation system.

        :param recommendations: A list of lists containing the recommended items for each query.
        :param ground_truth: A list containing the ground truth (relevant) items for each query.
        :return: The Mean Reciprocal Rank (MRR) value as a float.
        """
        assert len(recommendations) == len(ground_truth), "Recommendations and ground truth lists must have the same length."

        reciprocal_ranks = []

        for rec, gt in zip(recommendations, ground_truth):
            for rank, item in enumerate(rec, start=1):
                if item == gt:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mrr


    def _decoder(self, recommendation):

        '''decode sequeces to ASIN ids'''

        decoded = []
        for next_item in recommendation:

            decoded.append([self.items.iloc[e-1] for e in next_item])

        decoded = np.array(decoded)
        return decoded



    def _predictor(self, X_test):

        '''generate y_pred (which is top 100 product indices) from the model for X_test. '''

        batch_size = 64
        num_batches = int(len(X_test[0]) / batch_size)

        y_pred = []
        for batch_idx in range(num_batches+1):

            if batch_idx < num_batches:
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                inputs = []

                for i in range(len(X_test)):
                    inputs.append(X_test[i][start_idx:end_idx])

                predictions = self.model.predict(inputs)
                recom_size = 100


                top_preds = np.argpartition(predictions, -recom_size, axis=1)[:, -recom_size:]
                sorted_indices = np.argsort(predictions[np.arange(len(predictions))[:, None], top_preds], axis=1)[:, ::-1]
                recom = top_preds[np.arange(len(predictions))[:, None], sorted_indices]

                y_pred.append(recom)


            else:

                inputs = []

                for i in range(len(X_test)):

                    inputs.append(X_test[i][end_idx:])

                predictions = self.model.predict(inputs)

                top_preds = np.argpartition(predictions, -recom_size, axis=1)[:, -recom_size:]
                sorted_indices = np.argsort(predictions[np.arange(len(predictions))[:, None], top_preds], axis=1)[:, ::-1]
                recom = top_preds[np.arange(len(predictions))[:, None], sorted_indices]

                y_pred.append(recom)

        y_pred = [inner_list for outer_list in y_pred for inner_list in outer_list]

        return y_pred


    def test_1_testontest(self):


        '''evaluate model's performance on the test set defined in the initialization '''
        #update it for all the test sessions instead of only 200

        recommendation = self._predictor([self.X1_test, self.X2_test, self.X3_test, self.X4_test, self.X5_test, self.X6_test])
        gnd = self.y_test.tolist()
        self.test1_MRR = self._mean_reciprocal_rank(recommendation, gnd)
        print(f'MRR for test1: {self.test1_MRR}')




    def test_2_testwithendone(self, n=100):

        ''' evaluate model's performance on the given test set. Since test set has no ground truth
        we will split the last item in the session and consider it as the next item and evaulate model
        performance'''


        X1 = self.test_gdf['prev_items-list'].tolist()
        X2 = self.test_gdf['title-list'].tolist()
        X3 = self.test_gdf['brand-list'].tolist()
        X4 = self.test_gdf['size-list'].tolist()
        X5 = self.test_gdf['model-list'].tolist()
        X6 = self.test_gdf['color-list'].tolist()

        #handles variable length session sequences
        X1 = np.array(X1, dtype='object')


        #extract next item from the X1: prev_items_list, also remove last items attributes
        X1_p = []
        X2_p = []
        X3_p = []
        X4_p = []
        X5_p = []
        X6_p = []

        y_p = []

        for i in range(len(X1)):
            X1_p.append(X1[i][:-1])
            X2_p.append(X2[i][:-1])
            X3_p.append(X3[i][:-1])
            X4_p.append(X4[i][:-1])
            X5_p.append(X5[i][:-1])
            X6_p.append(X6[i][:-1])
            y_p.append(X1[i][-1])


        X1 = X1_p
        X2 = X2_p
        X3 = X3_p
        X4 = X4_p
        X5 = X5_p
        X6 = X6_p
        y= y_p
        y = np.array(y)

        #padding: pre for X1 and post for all others
        X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
        X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
        X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
        X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
        X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')


        rec = self._predictor([X1[:n], X2[:n], X3[:n], X4[:n], X5[:n], X6[:n]])
        gnd = y[:n].tolist()
        self.test2_MRR =  self._mean_reciprocal_rank(rec, gnd)
        print(f'MRR for test1: {self.test2_MRR}')




    def test_3_generatefinalresult(self, n=100):


        ''' generates predictions of test set. Decodes the index and return the recommendations with ASIN ids'''

        X1 = self.test_gdf['prev_items-list'].tolist()
        X2 = self.test_gdf['title-list'].tolist()
        X3 = self.test_gdf['brand-list'].tolist()
        X4 = self.test_gdf['size-list'].tolist()
        X5 = self.test_gdf['model-list'].tolist()
        X6 = self.test_gdf['color-list'].tolist()

        X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
        X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
        X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
        X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
        X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')

        rec = self._predictor([X1[:n], X2[:n], X3[:n], X4[:n], X5[:n], X6[:n]])

        y_pred = self._decoder(rec)
        y_pred = y_pred.tolist()
        df = pd.DataFrame()
        df['next_item_prediction'] = y_pred

        return df

