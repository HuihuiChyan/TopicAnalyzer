# 基于人民日报2023年数据的习近平讲话主题词分析

## 文件说明
* scrawler.py 爬虫程序，爬取人民日报文章
* retriever.py 对爬取的文章进行分析并可视化
* simfang.ttf 用于生成词云时设置中文字体
* figs/词云-主题模型.png 按照主题模型生成的主题词词云
* figs/词云-TFIDF.png 按照TFIDF算法生成的主题词词云
* figs/扇形图.png 对十个关键词在2023年全部文章中词频的扇形图
* figs/柱状图.png 对十个关键词在2023年全部文章中词频的柱状图
* figs/折线图.png 对某个关键词在2023年每个月份中词频变化的折线图

## 主题词提取与分析过程
### 基于TFIDF的主题词提取和词云生成
1. 首先，将2023年全部文章中包含“习近平”和“讲话”的文章都提取出来；
2. 然后使用jieba对所有文章进行分词；
3. 接着计算每个词的TFIDF值，从而对于每篇文章获取10个关键词；
4. 将所有文章的10个关键词合并到一起，取频次最高的前100个，得到基于TFIDF的关键词；
5. 基于wordcloud库，生成TFIDF关键词的词云；

### 基于主题模型的主题词提取和词云生成
1. 首先，将2023年全部文章中包含“习近平”和“讲话”的文章都提取出来；
2. 然后使用jieba对所有文章进行分词；
3. 接着训练主题模型，主题个数为10，每个主题用20个词表示；
4. 将所有主题的主题词合并到一起，得到基于主题模型的关键词；
5. 基于wordcloud库，生成主题模型关键词的词云；

### 代表词词频的扇形图和柱状图统计
1. 基于步骤二中产生的10个主题模型，对每个主题选取一个代表词；
2. 统计每个代表词在全部文章中的出现频次；
3. 基于matplotlib库，生成代表词词频的扇形图和柱状图统计；

### 代表词词频变化的折线图统计
1. 基于步骤二中产生的10个主题模型，选取几个代表性的关键词
2. 统计每个代表词在每个月份文章中的出现频次；
3. 基于matplotlib库，生成代表词词频随月份变化的折线图统计；