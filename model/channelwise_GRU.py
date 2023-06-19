import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# corresponding to model
import tensorflow as tf
from keras.layers import Input, Embedding, GRU, Dense, Concatenate, Dropout, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

class Metesre():

    def __init__(self):
        item_ids = pd.read_parquet('../preprocessing/categories/unique.prev_items.parquet')
        self.items = item_ids['prev_items']
        self.sessions_gdf = pd.read_parquet("../processed_nvt/part_0.parquet")
        self.test_gdf = pd.read_parquet("../processed_nvt/test_0.parquet")
        self.preprocessing()
        self.buildModel()


    def preprocessing(self):
        X1 = self.sessions_gdf['prev_items-list'].tolist()
        X2 = self.sessions_gdf['title-list'].tolist()

        #handles variable length session sequences
        X1 = np.array(X1, dtype='object')

        #find vocab sizes
        self.vocab_size1 = max(item for sublist in X1 for item in sublist)+1
        self.vocab_size2 = max(item for sublist in X2 for item in sublist)+1

        print("Vocab Sizes: \n",self.vocab_size1, self.vocab_size2)

        #extract next item from the X1: prev_items_list, also remove last items attributes
        X1_p = []
        X2_p = []

        y_p = []

        for i in range(len(X1)):
            X1_p.append(X1[i][:-1])
            X2_p.append(X2[i][:-1])
            y_p.append(X1[i][-1])

        X1 = X1_p
        X2 = X2_p
        y= np.array(y_p)

        # print(X1[0:5])
        # print(y[0:5])
        self.max_len = 10
        #padding: pre for X1 and post for all others
        X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')

        self.X1_train, self.X1_test, self.X2_train, self.X2_test, self.y_train, self.y_test = train_test_split(
            X1, X2, y, test_size=0.005, random_state=42, shuffle=True)

    def buildModel(self):

        embedding_dim = 128
        hidden_units = 48
        seq_length = self.max_len

        # Define the input layers
        input_layer1 = tf.keras.Input(shape=(seq_length,))
        input_layer2 = tf.keras.Input(shape=(seq_length,))

        # Define the embedding layers
        embedding_layer1 = Embedding(self.vocab_size1, embedding_dim)
        embedding_layer2 = Embedding(self.vocab_size2, embedding_dim)

        # Define the GRU layers
        gru_layer1 = GRU(hidden_units, return_sequences=False)
        gru_layer2 = GRU(hidden_units, return_sequences=False)

        # Define the dropout layer
        dropout_layer = Dropout(0.3)

        # Define the output layer
        output_layer = Dense(self.vocab_size1, activation='softmax')

        # Connect the layers
        embedded_input1 = embedding_layer1(input_layer1)
        embedded_input2 = embedding_layer2(input_layer2)

        # Apply the GRU layers with dropout to the embedded inputs
        gru_output1 = gru_layer1(embedded_input1)
        dropout_output1 = dropout_layer(gru_output1)

        gru_output2 = gru_layer2(embedded_input2)
        dropout_output2 = dropout_layer(gru_output2)


        # Concatenate the GRU outputs
        concatenated_output = tf.keras.layers.Concatenate()([dropout_output1, dropout_output2])

        output = output_layer(concatenated_output)

        # Create the model
        self.model = tf.keras.Model(inputs=[input_layer1, input_layer2], outputs=output)

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])

        # Print the model summary
        self.model.summary()

    def train(self, epoch= 10, batch_size = 32):
        checkpoint_callback = ModelCheckpoint(
            './model_checkpoint.h5',
            monitor='val_cosine_similarity',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        self.history = self.model.fit(
            [self.X1_train, self.X2_train],
            self.y_train,
            epochs = epoch,
            batch_size=batch_size,
            verbose = True,
            validation_split=0.1,
            callbacks=[checkpoint_callback]
        )

    def _accuracy(self, predictions, ground_truth):
        correct = 0
        total = len(ground_truth)
        for pred, label in zip(predictions, ground_truth):
            for rank, item in enumerate(pred, start=1):
                if item == label:
                    correct += 1
        accuracy = correct / total
        return accuracy

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

        recommendation = self._predictor([self.X1_test, self.X2_test])
        gnd = self.y_test.tolist()
        self.test1_MRR = self._mean_reciprocal_rank(recommendation, gnd)
        print(f'MRR for test1: {self.test1_MRR}')




    def test_2_testwithendone(self, n=100):

        ''' evaluate model's performance on the given test set. Since test set has no ground truth
        we will split the last item in the session and consider it as the next item and evaulate model
        performance'''


        X1 = self.test_gdf['prev_items-list'].tolist()
        X2 = self.test_gdf['title-list'].tolist()

        #handles variable length session sequences
        X1 = np.array(X1, dtype='object')


        #extract next item from the X1: prev_items_list, also remove last items attributes
        X1_p = []
        X2_p = []

        y_p = []

        for i in range(len(X1)):
            X1_p.append(X1[i][:-1])
            X2_p.append(X2[i][:-1])
            y_p.append(X1[i][-1])


        X1 = X1_p
        X2 = X2_p
        y= y_p
        y = np.array(y)

        #padding: pre for X1 and post for all others
        X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')

        rec = self._predictor([X1[:n], X2[:n]])
        gnd = y[:n].tolist()
        self.test2_MRR =  self._mean_reciprocal_rank(rec, gnd)
        print(f'MRR for test2: {self.test2_MRR}')




    def test_3_generatefinalresult(self, n=100):


        ''' generates predictions of test set. Decodes the index and return the recommendations with ASIN ids'''

        X1 = self.test_gdf['prev_items-list'].tolist()
        X2 = self.test_gdf['title-list'].tolist()

        X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
        X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')

        rec = self._predictor([X1[:n], X2[:n]])

        y_pred = self._decoder(rec)
        y_pred = y_pred.tolist()
        df = pd.DataFrame()
        df['next_item_prediction'] = y_pred

        return df


