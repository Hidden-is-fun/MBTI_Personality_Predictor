import random

import bert
import pandas as pd
import tensorflow_hub as hub

import predict
import process
import tensorflow as tf

if __name__ == "__main__":
    data_set = pd.read_csv("data/mbti.csv")
    text = []
    types = []
    _ = []
    for _i in range(10):
        text.append(data_set["posts"][_i])
        types.append(data_set["type"][_i])

    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    for _i in range(10):
        _.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[_i])))

    for _i in range(10):
        if len(_[_i]) > 2000:
            _[_i] = _[_i][:2000]
        for __ in range(2000 - len(_[_i])):
            _[_i].append(0)

    loaded = tf.keras.models.load_model("FT")
    for __ in range(10):
        print(types[__], end=' ')
        a = loaded.predict([_[__]])
        print('%.4f' % a[0][0])
        print(a)
    exit()

    cntizer = predict.Initialize().load_count_vectorizer()
    tfizer = predict.Initialize().load_tfidf_transformer()
    my_posts = input()
    a, b = predict.Predict(my_posts, cntizer, tfizer).predict()

    personality_type = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)",
                        "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"]

    data_set, types = process.load_dataset()
    data_set = process.PreProcess(data_set).add_intj_row()
    list_posts, list_personality = process.PreProcess(data_set).pre_process_text(1, 1)
    cntizer = process.FeatureEngineering(list_posts).cntizer
    tfizer = process.FeatureEngineering(list_posts).tfizer
    X = process.FeatureEngineering(list_posts).X_tfidf
    print(X[1])
    print(list_personality[:, 1])
    exit(1)
    predict.FitAndPredict(cntizer, tfizer, list_personality, X)
    exit(3)
