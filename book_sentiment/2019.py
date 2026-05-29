import pandas as pd
import matplotlib.pyplot as plt
import jieba
import re
from wordcloud import WordCloud
from collections import Counter

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据读取与合并 ====================
meta = pd.read_csv('douban_feminism_books_metadata.csv', encoding='utf-8')
comments = pd.read_csv('douban_books_comments_with_sentiment.csv', encoding='utf-8')

# 合并
df = comments.merge(meta[['书名', '出版年']], on='书名', how='left')
df['出版年'] = pd.to_numeric(df['出版年'], errors='coerce')
df = df.dropna(subset=['出版年']).astype({'出版年': int})

# 确认文本列和情感列
text_col = '评论内容'
sent_label_col = 'sentiment_label'
sent_score_col = 'sentiment_score' if 'sentiment_score' in df.columns else None

# ==================== 2. 女性议题标记 ====================
female_keywords = [
    '她', '女人', '女性', '妇女', '女儿', '母亲', '妈妈', '妻子', '老婆',
    '婚姻', '结婚', '离婚', '生育', '生子', '子宫', '月经',
    '性别', '女性主义', '女权', '女拳', '直男癌', '厌女', '父权',
    '身材', '容貌', '穿衣自由', '化妆', '独立女性',
    '家庭主妇', '全职妈妈', '母职', '催婚', '相亲', '剩女'
]
df['female_related'] = df[text_col].apply(
    lambda x: any(kw in str(x) for kw in female_keywords)
)

# ==================== 3. 锁定2019年数据 ====================
df_2019 = df[df['出版年'] == 2019]
df_2019_f = df_2019[df_2019['female_related']]

print(f"2019年总评论数: {len(df_2019)}")
print(f"2019年女性议题评论数: {len(df_2019_f)}")
print(f"女性议题占比: {len(df_2019_f)/len(df_2019):.2%}")

# ==================== 4. 情感均值对比 ====================
if sent_score_col:
    mean_all_2019 = df_2019[sent_score_col].mean()
    mean_f_2019 = df_2019_f[sent_score_col].mean()
else:
    sent_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    mean_all_2019 = df_2019[sent_label_col].map(sent_map).mean()
    mean_f_2019 = df_2019_f[sent_label_col].map(sent_map).mean()

print(f"\n2019年全部评论情感均值: {mean_all_2019:.4f}")
print(f"2019年女性议题评论情感均值: {mean_f_2019:.4f}")

# ==================== 5. 分离负面评论（用于深挖） ====================
if sent_score_col:
    negative_f = df_2019_f[df_2019_f[sent_score_col] < 0]
    negative_f = negative_f.sort_values(sent_score_col)
else:
    negative_f = df_2019_f[df_2019_f[sent_label_col] == 'negative']

print(f"\n2019年女性议题负面评论数: {len(negative_f)}")

# ==================== 6. 负面评论高频词分析 ====================
# 加载停用词（使用常见中文停用词表，或手动定义简单版）
stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
                 '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
                 '没有', '看', '好', '自己', '这', '她', '他', '它', '们', '那', '但',
                 '可以', '这个', '那个', '什么', '怎么', '因为', '所以', '如果', '还是',
                 '觉得', '可能', '知道', '里面', '出来', '一样', '时候', '然后', '不能',
                 '不是', '就是', '比较', '这么', '那么', '一点', '有点', '一些', '很多',
                 '还是', '不过', '之后', '以前', '只能', '只是', '的话', '啊', '吧',
                 '哦', '嗯', '呢', '吗', '呀', '啦', '噢', '哟','已经','他们','应该','我们','非常','但是','女性','其实','这样','如何','我们','本书'])

def get_top_words(text_series, topn=30):
    word_list = []
    for text in text_series:
        words = jieba.cut(str(text))
        word_list.extend([w for w in words if w not in stopwords and len(w) > 1])
    return Counter(word_list).most_common(topn)

top_neg_words = get_top_words(negative_f[text_col])
print("\n=== 2019年女性议题负面评论高频词 TOP30 ===")
for w, c in top_neg_words:
    print(f"{w}: {c}")

# ==================== 7. 展示极端负面评论样本 ====================
print("\n=== 2019年女性议题中最负面的5条评论 ===")
if sent_score_col:
    for i, row in negative_f.head(5).iterrows():
        print(f"[得分: {row[sent_score_col]:.4f}] {row[text_col][:200]}")
else:
    for i, row in negative_f.head(5).iterrows():
        print(f"[-] {row[text_col][:200]}")

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ==================== 8. 升级版词云：只保留社会议题相关词汇 ====================

# 8.1 定义强效停用词表（书评套路词 + 通用停用词）
stopwords = set([
    # 原通用词
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '她', '他', '它', '们', '那', '但',
    '可以', '这个', '那个', '什么', '怎么', '因为', '所以', '如果', '还是',
    '觉得', '可能', '知道', '里面', '出来', '一样', '时候', '然后', '不能',
    '不是', '就是', '比较', '这么', '那么', '一点', '有点', '一些', '很多',
    '还是', '不过', '之后', '以前', '只能', '只是', '的话', '啊', '吧',
    '哦', '嗯', '呢', '吗', '呀', '啦', '噢', '哟',
    # 书评/评论常见无意义词
    '本书', '作者', '读者', '文字', '阅读', '看过', '看完', '本书', '一部',
    '故事', '小说', '作品', '书中', '感觉', '真的', '特别', '完全', '比较',
    '可能', '其实', '有点', '算是', '部分', '整体', '内容', '情节', '人物',
    '描写', '看到', '感受', '理解', '方式', '讲', '写', '读', '让', '把',
    '被', '中', '更', '太', '还', '很', '大', '多', '年', '天', '时间',
    '没', '吧', '啊', '呀', '呢', '么', '这', '那', '它', '他', '我', '你','女性','需要','发生','应该','读完','女性主义','金智英','无法','事实','走向','要求','过程','东西','重写','事情','喜欢','实在','起来','试图','反应','山口','真琴','细节','序言','世界','获得',
])

# 8.2 自定义议题增强词表（这些词即便TF‑IDF值稍低也会保留）
boost_words = set([
    '婚育', '母职', '父权', '厌女', '性侵', '性骚扰', '家暴', '重男轻女',
    '剩女', '催婚', '相亲', '生育', '子宫', '月经', '身体', '身材', '容貌',
    '穿衣自由', '独立女性', '全职妈妈', '家庭主妇', '职场歧视', '性同意',
    '荡妇', '羞辱', '压迫', '觉醒', '共情', '恐惧', '焦虑', '压抑',
    '暴力', '凝视', '物化', '规训', '羞耻', '歧视', '平等', '权利',
])

# 8.3 使用 TF‑IDF 提取负面评论中的关键术语（只留名词/动词）
# 先对每条负面评论做分词和词性过滤
import jieba.posseg as pseg


def extract_nouns_verbs(text):
    words = pseg.cut(text)
    # 保留名词("n")、名动词("vn")、动词("v")，排除副词、形容词等
    allowed_flags = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn'}
    return ' '.join([w.word for w in words if w.flag in allowed_flags and w.word not in stopwords and len(w.word) > 1])


# 构建语料
corpus = negative_f[text_col].astype(str).apply(extract_nouns_verbs).tolist()

# 如果有效语料太少，降级为普通词频
if sum(len(doc.split()) for doc in corpus) < 50:
    print("有效词数过少，使用增强停用词词频模式")
    word_counter = Counter()
    for doc in corpus:
        word_counter.update(doc.split())
    # 加权提升议题词
    for w in word_counter:
        if w in boost_words:
            word_counter[w] *= 3
    wordcloud_dict = dict(word_counter)
else:
    # TF‑IDF 向量化
    vectorizer = TfidfVectorizer(max_features=200, max_df=0.8, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    # 计算每个词的平均 TF‑IDF 得分
    tfidf_means = np.mean(tfidf_matrix, axis=0).A1
    word_scores = dict(zip(feature_names, tfidf_means))

    # 提升议题关键词权重
    for w in word_scores:
        if w in boost_words:
            word_scores[w] *= 3.0
        elif w in stopwords:
            word_scores[w] = 0.0

    # 归一化并转为整数频次以适配 WordCloud
    max_score = max(word_scores.values()) if word_scores else 1
    wordcloud_dict = {w: int(s / max_score * 1000) for w, s in word_scores.items() if s > 0}

# 8.4 生成词云（直接用权重字典）
if wordcloud_dict:
    wc = WordCloud(
        font_path='simhei.ttf',
        width=1000,
        height=700,
        background_color='white',
        colormap='Reds',
        max_words=150
    ).generate_from_frequencies(wordcloud_dict)

    plt.figure(figsize=(12, 9))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('2019年女性议题负面书评的社会议题词云（TF‑IDF + 词性过滤）', fontsize=18)
    plt.tight_layout()
    plt.savefig('2019_negative_issue_wordcloud.png', dpi=300)
    plt.show()
else:
    print("无可用词生成词云")