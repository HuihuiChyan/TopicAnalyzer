import jieba
import os
import datetime
import time
from tqdm import tqdm
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

def gen_dates(b_date, days):
    day = datetime.timedelta(days = 1)
    for i in range(days):
        yield b_date + day * i

def get_date_list(beginDate, endDate):
    """
    获取日期列表
    :param start: 开始日期
    :param end: 结束日期
    :return: 开始日期和结束日期之间的日期列表
    """

    start = datetime.datetime.strptime(beginDate, "%Y%m%d")
    end = datetime.datetime.strptime(endDate, "%Y%m%d")

    data = []
    for d in gen_dates(start, (end-start).days):
        data.append(d)

    return data

def load_articles(path):
    paths = os.listdir(path)

    selected_articles = []

    for article in paths:
        with open(os.path.join(path, article), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            lines = "\n".join(lines)
            if "习近平" in lines and "讲话" in lines:
                selected_articles.append(lines)
    
    return selected_articles

def get_articles(beginDate, endDate):
    data = get_date_list(beginDate, endDate)

    all_articles = []
    for d in tqdm(data):
        year = str(d.year)
        month = str(d.month) if d.month >=10 else '0' + str(d.month)
        day = str(d.day) if d.day >=10 else '0' + str(d.day)

        path = f"{year}/{year}{month}{day}"

        all_articles.extend(load_articles(path))
    
    return all_articles

def get_topic_words_tfidf(seg_articles):

    cat_articles = [" ".join(article) for article in seg_articles]

    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform(cat_articles)

    idf = tf.idf_ 

    with open("idf.txt", "w", encoding="utf-8") as fout:
        for word in list(tf.vocabulary_.keys()):
            fout.write(word+" "+str(float(tf.idf_[tf.vocabulary_[word]]))+"\n")

    jieba.analyse.set_idf_path("idf.txt")
    topics_words = []
    for article in tqdm(all_articles):
        topics_words.extend(jieba.analyse.extract_tags(article, topK=10, withWeight=False, allowPOS=("n")))
    
    counter = collections.Counter(topics_words)
    topic_words = [word[0] for word in list(counter.most_common()[2:102])]

    return topic_words

def get_topic_words_topic_model(seg_articles):

    dictionary = gensim.corpora.Dictionary(seg_articles)
    dictionary.filter_extremes(no_below=3)
    corpus = [dictionary.doc2bow(article) for article in seg_articles]

    num_topics = 20
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, \
                                                id2word=dictionary, \
                                                passes=4, alpha=[0.01]*num_topics, \
                                                eta=[0.01]*len(dictionary.keys()))

    topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    topic_words = []
    for topic in topics:
        topic_words.extend([wd[0] for wd in topic[1]])

    return topic_words

def build_word_cloud(d, cloud_name):

    wordcloud = WordCloud(background_color="white",
                          height=800,
                          width=1600,
                          font_path='./simfang.ttf')
    wordcloud.generate_from_frequencies(frequencies=d)

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"figs/{cloud_name}.png")

if __name__ == '__main__':

    beginDate = "20230101"
    endDate = "20231231"
    all_articles = get_articles(beginDate, endDate)

    all_articles = list(set(all_articles))   
    seg_articles = [[word for word in jieba.cut(article, cut_all=True)] for article in all_articles]

    seg_articles_concat = " ".join([" ".join(article) for article in seg_articles]).split()
    counter = collections.Counter(seg_articles_concat)

    # 获取基于TFIDF的主题词
    topic_words = get_topic_words_tfidf(seg_articles)

    # 统计主题词的词频并生成词云
    d = {}
    for word in topic_words:
        d[word] = counter[word]
    
    build_word_cloud(d, cloud_name="词云-TFIDF")

    # 获取基于Topic Model的主题词
    topic_words = get_topic_words_topic_model(seg_articles)

    # 统计主题词的词频并生成词云
    d = {}
    for word in topic_words:
        d[word] = counter[word]
    
    build_word_cloud(d, cloud_name="词云-主题模型")


    # 对于每个类别，挑出一个词，并对其进行统计
    words = ["中华文明", "从严治党", "二十大", "疫情", "一带", "香港", "生态环境", "民营企业", "马克思主义", "金融"]

    freqs = []
    for word in words:
        freqs.append(counter[word])
    
    plt.rcParams['font.family'] = 'Heiti TC' 
    
    words = np.array(words)
    freqs = np.array(freqs)

    plt.figure(figsize=(10,6))
    plt.bar(words, freqs)
    plt.savefig("figs/扇形图.png")

    fig, ax = plt.subplots()
    ax.pie(freqs, labels=words)
    plt.savefig("figs/柱状图.png")


    # 对于每个关键词，统计其按照月份的出现频次
    words = ["二十大", "疫情", "香港", "一带"]
    for word in words:
        months = []
        freqs = []
        for month in range(1, 13):
            if month < 10:
                month = f"0{month}"
            else:
                month = str(month)

            beginDate = f"2023{month}01"
            if month != "12":
                next_month = str(int(month) + 1)
                endDate = f"2023{next_month}01"
            else:
                endDate = f"20231231"

            all_articles = get_articles(beginDate, endDate)

            all_articles = list(set(all_articles))
            seg_articles = [[word for word in jieba.cut_for_search(article)] for article in all_articles]

            seg_articles_concat = " ".join([" ".join(article) for article in seg_articles]).split()

            counter = collections.Counter(seg_articles_concat)

            months.append(month)
            freqs.append(counter[word])

        plt.clf()
        plt.plot(months, freqs)  # Plot the chart
        plt.savefig(f"figs/折线图-{word}.png")