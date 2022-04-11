import pickle
import random
import time

import bert
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import process
import tensorflow_hub as hub
import tensorflow as tf


class FitAndPredict:
    # Fit and train using TF-IDF + XGBoost method
    def __init__(self, cntizer, tfizer, list_personality=None, X=None):
        my_posts = """They act like they care They tell me to share But when I carve the stories on my arm The doctor 
        just calls it self harm I’m not asking for attention There’s a reason I have apprehensions I just need you to 
        see What has become of me||| I know I’m going crazy But they think my thoughts are just hazy When in that 
        chaos, in that confusion I’m crying out for help, to escape my delusions||| Mental health is a state of mind 
        How does one keep that up when assistance is denied All my failed attempts to fight the blaze You treat it 
        like its a passing phase||| Well stop, its not, because mental illness is real Understand that we’re all not 
        made of steel Because when you brush these issues under the carpet You make it seem like its our mistake 
        we’re not guarded||| Don’t you realise that its a problem that needs to be addressed Starting at home, 
        in our nest Why do you keep your mouths shut about such things Instead of caring for those with broken 
        wings||| What use is this social stigma When mental illness is not even such an enigma Look around and you’ll 
        see the numbers of the affected hiding under the covers ||| This is an issue that needs to be discussed Not 
        looked down upon with disgust Mental illness needs to be accepted So that people can be protected ||| Let me 
        give you some direction People need affection The darkness must be escaped Only then the lost can be saved||| 
        Bring in a change Something not very strange The new year is here Its time to eradicate fear||| Recognise the 
        wrists under the knives To stop mental illness from taking more lives Let’s break the convention Start 
        ‘suicide prevention’.||| Hoping the festival of lights drives the darkness of mental illness away """
        mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

        my_posts, dummy = process.PreProcess(mydata).pre_process_text(1, 1)
        print(my_posts)

        my_X_cnt = cntizer.transform(my_posts)

        print(my_X_cnt)
        my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

        # XGBoost model for MBTI dataset
        result = []
        personality_type = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)",
                            "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"]
        # Individually training each mbti personlity type
        '''
        for l in range(len(main.personality_type)):
            model = xgboost.Booster(model_file=f"{personality_type[l][:2]}.model")
            y_pred = model.predict(xgboost.DMatrix(my_X_tfidf))
            result.append(y_pred[0])
        '''
        for l in range(len(personality_type)):
            print("%s classifier trained" % (personality_type[l]))

            Y = list_personality[:, l]

            # split data into train and test sets
            seed = 7
            test_size = 0.33
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

            param = {}
            param['n_estimators'] = 200
            param['max_depth'] = 2
            param['nthread'] = 8
            param['learning_rate'] = 0.2

            # fit model on training data
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)
            joblib.dump(model, f'models/{personality_type[l][:2]}.model')

            # make predictions for my  data
            y_pred = model.predict(my_X_tfidf)
            result.append(y_pred)

        print(result)

        # print("The result is: ", process.PreProcess([]).translate_back(result))


class Initialize:
    def __init__(self):
        pass

    @staticmethod
    def load_count_vectorizer():
        feature_path = 'models/feature.pkl'
        cntizer = CountVectorizer(analyzer="word",
                                  max_features=1000,
                                  max_df=0.7,
                                  min_df=0.1,
                                  vocabulary=pickle.load(open(feature_path, "rb")))
        time.sleep(random.randint(1, 10))
        return cntizer

    @staticmethod
    def load_tfidf_transformer():
        tfidftransformer_path = 'models/tfidftransformer.pkl'
        tfizer = pickle.load(open(tfidftransformer_path, "rb"))
        time.sleep(random.randint(1, 10))
        return tfizer

    @staticmethod
    def load_tokenizer():
        BertTokenizer = bert.bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
        vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
        return tokenizer

    @staticmethod
    def load_model(id):
        personality_type = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)",
                            "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"]
        model = tf.keras.models.load_model(f'models/{personality_type[id][:2]}')
        return model


class Predict:
    def __init__(self, my_posts, cntizer, tfizer):
        mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})
        self.my_posts, dummy = process.PreProcess(mydata).pre_process_text(1, 1)

        my_X_cnt = cntizer.transform(self.my_posts)
        self.my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

    def predict(self):
        result = ''
        personality_type = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)",
                            "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"]
        type_chart = []
        score_chart = []
        for _i in range(4):
            lr = joblib.load(f'models/{personality_type[_i][:2]}.model')
            res = lr.predict_proba(self.my_X_tfidf)
            score = float(str(res[0])[1:8]) * 100
            score_chart.append(score)
            type_chart.append(False if score <= 50 else True)
            result += f'\033[31m{personality_type[_i][0]}\033[0m: '
            result += '\033[31m' if score > 50 else '\033[34m'
            result += f'{str(score)[:5]}%'
            result += f'\033[0m /\033[34m{personality_type[_i][1]}\n'
            # result += personality_type[_i][1] if res else personality_type[_i][0]
        result += '\033[0mThe result is: '
        for _i in range(4):
            result += '\033[31m' if type_chart[_i] else '\033[34m'
            result += f'{personality_type[_i][0] if type_chart[_i] else personality_type[_i][1]}'
        print(result)
        return type_chart, score_chart

    def words(self):
        return self.my_posts[0]


class PredictWithBERT:
    def __init__(self, my_posts, token, model):
        self.my_posts = my_posts
        self.token = token
        self.model = model

    def encoding(self):
        _: list = self.token.convert_tokens_to_ids(self.token.tokenize(self.my_posts))
        if len(_) > 2000:
            _ = _[:2000]
        for __ in range(2000 - len(_)):
            _.append(0)
        return _

    def predict(self):
        type_chart = []
        result = []
        encoding = self.encoding()
        for _ in range(4):
            a = self.model[_].predict([encoding])
            print(a)
            type_chart.append(True if a[0][0] >= 0.5 else False)
            _res = '%.3f' % (a[0][0] * 100)
            result.append(float(_res))
        return type_chart, result



