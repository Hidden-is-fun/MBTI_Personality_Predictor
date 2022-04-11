import pickle

import numpy as np
import pandas as pd
import re

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset():
    data_set = pd.read_csv("data/mbti.csv")
    types = np.unique(np.array(data_set['type']))
    return data_set, types


def preprocess_text(df, remove_special=True):
    texts = df['posts'].copy()
    labels = df['type'].copy()

    # Remove links
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|", " ") + " "))

    # Keep the End Of Sentence characters
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # Strip Punctuation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[\.+]', ".", x))

    # Remove multiple fullstops
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove Non-words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Convert posts to lowercase
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    # Remove multiple letter repeating words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x))

    # Remove very short or long words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))

    # Remove MBTI Personality Words - crucial in order to get valid model accuracy estimation for unseen data.
    if remove_special:
        pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ',
                      'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    return df


def remove_short_posts(l, df):
    # print("Before : Number of posts", len(df))
    df["no. of. words"] = df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
    df = df[df["no. of. words"] >= l]
    # print("After : Number of posts", len(df))
    return df


class PreProcess:
    def __init__(self, data):
        self.data = data
        self.lemmatiser = WordNetLemmatizer()
        self.unique_type_list = self.get_unique_type()
        # Remove the stop words for speed
        self.useless_words = stopwords.words("english")
        self.b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
        self.b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]

    @staticmethod
    def get_types(row):
        t = row['type']

        I = 0
        N = 0
        T = 0
        J = 0

        if t[0] == 'I':
            I = 1
        elif t[0] == 'E':
            I = 0
        else:
            print('I-E not found')

        if t[1] == 'N':
            N = 1
        elif t[1] == 'S':
            N = 0
        else:
            print('N-S not found')

        if t[2] == 'T':
            T = 1
        elif t[2] == 'F':
            T = 0
        else:
            print('T-F not found')

        if t[3] == 'J':
            J = 1
        elif t[3] == 'P':
            J = 0
        else:
            print('J-P not found')
        return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})

    # Translate 'INTJ'-like data to binary
    def add_intj_row(self):
        self.data = self.data.join(self.data.apply(lambda row: self.get_types(row), axis=1))
        return self.data

    @staticmethod
    def get_unique_type():
        unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                            'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
        unique_type_list = [x.lower() for x in unique_type_list]
        return unique_type_list

    def translate_personality(self, personality):
        # transform mbti to binary vector
        return [self.b_Pers[l] for l in personality]

    # To show result output for personality prediction
    def translate_back(self, personality):
        # transform binary vector to mbti personality
        s = ""
        for i, l in enumerate(personality):
            s += self.b_Pers_list[i][l]
        return s

    def pre_process_text(self, remove_stop_words=True, remove_mbti_profiles=True):
        list_personality = []
        list_posts = []
        len_data = len(self.data)
        i = 0

        for row in self.data.iterrows():
            # Remove and clean comments
            posts = row[1].posts

            # Remove url links
            temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

            # Remove Non-words - keep only words
            temp = re.sub("[^a-zA-Z]", " ", temp)

            # Remove spaces > 1
            temp = re.sub(' +', ' ', temp).lower()

            # Remove multiple letter repeating words
            temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

            # Remove stop words
            if remove_stop_words:
                temp = " ".join([self.lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in self.useless_words])
            else:
                temp = " ".join([self.lemmatiser.lemmatize(w) for w in temp.split(' ')])

            # Remove MBTI personality words from posts
            if remove_mbti_profiles:
                for t in self.unique_type_list:
                    temp = temp.replace(t, "")

            # transform mbti to binary vector
            type_labelized = self.translate_personality(row[1].type)  # or use lab_encoder.transform([row[1].type])[0]
            list_personality.append(type_labelized)
            # the cleaned data temp is passed here
            list_posts.append(temp)

        # returns the result
        list_posts = np.array(list_posts)
        list_personality = np.array(list_personality)
        return list_posts, list_personality


class FeatureEngineering:
    def __init__(self, posts):
        self.list_posts = posts
        self.cntizer = CountVectorizer(analyzer="word",
                                       max_features=1000,
                                       max_df=0.7,
                                       min_df=0.1,)
        self.X_cnt = self.cntizer.fit_transform(self.list_posts)

        self.tfizer = TfidfTransformer()
        self.X_tfidf = self.tfizer.fit_transform(self.X_cnt).toarray()

        # print(self.cntizer.vocabulary_)
        # print(self.tfizer)

        '''
        # save feature and tf-idf model
        feature_path = 'models/feature.pkl'
        with open(feature_path, 'wb') as fw:
            pickle.dump(self.cntizer.vocabulary_, fw)

        tfidf_path = 'models/tfidftransformer.pkl'
        with open(tfidf_path, 'wb') as fw:
            pickle.dump(self.tfizer, fw)
        '''


