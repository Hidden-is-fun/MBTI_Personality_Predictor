import collections
import wordcloud
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import warnings


warnings.filterwarnings("ignore")

font = FP(fname="msyh.ttc", size=15)


#  显示各性格类型数量
def post_count(data_set):
    cnt_srs = data_set['type'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
    plt.xlabel("性格类型", fontproperties=font)
    plt.ylabel("帖文数量", fontproperties=font)
    plt.show()


#  显示蜜蜂图
def swarm_plot(df):
    def var_row(row):
        l = []
        for i in row.split('|||'):
            l.append(len(i.split()))
        return np.var(l)

    df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split()) / 50)
    df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))

    plt.figure(figsize=(15, 12))
    sns.swarmplot("type", "words_per_comment", data=df)
    plt.xlabel("性格类型", fontproperties=font)
    plt.ylabel("平均发帖长度（单词/帖文）", fontproperties=font)
    plt.show()


#  显示词频最高词语
def common_word(df):
    words = list(df["posts"].apply(lambda x: x.split()))
    words = [x for y in words for x in y]
    _common = collections.Counter(words).most_common(50)
    print(" ")
    for _i in range(50):
        print('%-15s%-10s' % (_common[_i][0], _common[_i][1]))


#  词云
def word_cloud(data_set):
    words = list(data_set["posts"].apply(lambda x: x.split()))
    words = [x for y in words for x in y]
    wc = wordcloud.WordCloud(width=1000, height=1000,
                             collocations=False, background_color="white").generate(" ".join(words))
    plt.figure(figsize=(10, 10))
    plt.imshow(wc)
    _ = plt.axis("off")
    plt.show()


#  INTJ四维比例
def show_INTJ(data):
    print("Introversion (I) / Extroversion (E):   ", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
    print("Intuition (N) / Sensing (S):           ", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
    print("Thinking (T) / Feeling (F):            ", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
    print("Judging (J) / Perceiving (P):          ", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])


def show_INTJ_plot(data):
    N = 4
    bottom = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0],
              data['JP'].value_counts()[0])
    top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1],
           data['JP'].value_counts()[1])

    ind = np.arange(N)  # the x locations for the groups
    # the width of the bars
    width = 0.7  # or len(x) can also be used here

    p1 = plt.bar(ind, bottom, width, label="I, N, T, F")
    p2 = plt.bar(ind, top, width, bottom=bottom, label="E, S, F, P")

    plt.title('Distribution accoss types indicators')
    plt.ylabel('Count')
    plt.xticks(ind, ('I / E', 'N / S', 'T / F', 'J / P',))
    plt.legend()

    plt.show()