import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from tensorflow.keras import layers
import bert
import pandas as pd
import numpy as np
import re
import random
import math
import os


# INFJ/ESTP

class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


def tokenize_text(text_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_input))


if __name__ == '__main__':

    # hyper parameters
    BATCH_SIZE = 128
    EMB_DIM = 300
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 10
    DROPOUT_RATE = 0.5
    NB_EPOCHS = 20
    max_len = 2000

    # raw data

    data_set = pd.read_csv("data/mbti.csv")
    y_4axis = [[], [], [], []]
    text = []
    personality_type = ['IE', 'NS', 'FT', 'JP']
    for _i in range(len(data_set)):
        _text = data_set["posts"][_i]
        _text = _text[1:-1]
        _text = re.sub(r'https?:\/\/.*?[\s+]', ' ', _text)
        _text = re.sub(r'http?:\/\/.*?[\s+]', ' ', _text)
        _text = _text.replace('...|||', ' ')
        _text = _text.replace('|||', ' ')
        text.append(_text)
        for _ in range(4):
            y_4axis[_].append(0 if data_set["type"][_i][_] == personality_type[_][0] else 1)

    # Creating a BERT Tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # Tokenize all the text
    tokenized_text = [tokenize_text(i) for i in text]

    for _i in range(4):
        # Prerparing Data For Training
        text_with_len = [[text, y_4axis[_i][i], len(text)]
                         for i, text in enumerate(tokenized_text)]
        random.shuffle(text_with_len)
        # text_with_len.sort(key=lambda x: x[2])
        # sorted_text_labels = [(text_lab[0], text_lab[1]) for text_lab in text_with_len]
        sorted_text_labels = [(text_lab[0][:max_len], text_lab[1]) for text_lab in text_with_len]
        processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_text_labels,
                                                           output_types=(tf.int32, tf.int32))

        # batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
        batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((max_len,), ()))

        TOTAL_BATCHES = math.ceil(len(sorted_text_labels) / BATCH_SIZE)
        TEST_BATCHES = TOTAL_BATCHES // 20
        batched_dataset.shuffle(TOTAL_BATCHES)
        test_data = batched_dataset.take(TEST_BATCHES)
        train_data = batched_dataset.skip(TEST_BATCHES)

        VOCAB_LENGTH = len(tokenizer.vocab)
        text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                                embedding_dimensions=EMB_DIM,
                                cnn_filters=CNN_FILTERS,
                                dnn_units=DNN_UNITS,
                                model_output_classes=OUTPUT_CLASSES,
                                dropout_rate=DROPOUT_RATE)

        if OUTPUT_CLASSES == 2:
            text_model.compile(loss="binary_crossentropy",
                               optimizer="adam",
                               metrics=["accuracy"])
        else:
            text_model.compile(loss="sparse_categorical_crossentropy",
                               optimizer="adam",
                               metrics=["sparse_categorical_accuracy"])

        text_model.fit(train_data, epochs=NB_EPOCHS)
        # text_model.fit(train_data, epochs=NB_EPOCHS,validation_data=test_data)
        # test test data
        results = text_model.evaluate(test_data)
        print(f'{personality_type[_i][0]}/{personality_type[_i][1]} Trained Successfully!\n Accuracy: {results[1] * 100}%')
        text_model.save(f'{personality_type[_i][0]}{personality_type[_i][1]}')
        print(f'model {personality_type[_i][0]}/{personality_type[_i][1]} saved.\n\n')
