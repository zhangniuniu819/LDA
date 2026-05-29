import pandas as pd
import matplotlib.pyplot as plt
import jieba.posseg as pseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 文件路径 ====================
META_FILE = 'doulist_feminism_movies.csv'
COMMENTS_FILE = 'douban_film_comments_with_sentiment.csv'

# ==================== 1. 读取数据 ====================
meta = pd.read_csv(META_FILE, encoding='utf-8')
comments = pd.read_csv(COMMENTS_FILE, encoding='utf-8')

# 列名（与你的数据一致）
META_NAME = '电影名'
META_YEAR = '上映年份'
COM_NAME = '电影名'
COM_TEXT = '评论内容'
COM_LABEL = 'sentiment_label'

# ==================== 2. 合并 + 锁定2020年 ====================
df = comments.merge(meta[[META_NAME, META_YEAR]], on=COM_NAME, how='left')
df[META_YEAR] = pd.to_numeric(df[META_YEAR], errors='coerce')
df = df.dropna(subset=[META_YEAR])
df[META_YEAR] = df[META_YEAR].astype(int)

# 仅2020年
df_2020 = df[df[META_YEAR] == 2020].copy()
print(f"2020年电影评论总数：{len(df_2020)}")

# ==================== 3. 女性议题标记 ====================
female_keywords = [
    '她', '女人', '女性', '妇女', '女儿', '母亲', '妈妈', '妻子', '老婆',
    '婚姻', '结婚', '离婚', '生育', '生子', '子宫', '月经',
    '性别', '女性主义', '女权', '女拳', '直男癌', '厌女', '父权',
    '身材', '容貌', '穿衣自由', '化妆', '独立女性',
    '家庭主妇', '全职妈妈', '母职', '催婚', '相亲', '剩女'
]

df_2020['female_related'] = df_2020[COM_TEXT].apply(
    lambda x: any(kw in str(x) for kw in female_keywords)
)
df_2020_f = df_2020[df_2020['female_related']]
print(f"其中女性议题评论：{len(df_2020_f)}")

# ==================== 4. 提取负面评论 ====================
if 'sentiment_score' in df_2020_f.columns:
    negative_f = df_2020_f[df_2020_f['sentiment_score'] < 0]
    negative_f = negative_f.sort_values('sentiment_score')
else:
    negative_f = df_2020_f[df_2020_f[COM_LABEL] == 'negative'].copy()

print(f"负面女性议题评论数：{len(negative_f)}")

# ==================== 5. 定义强效停用词表 ====================
stopwords = set([
    # 通用
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '她', '他', '它', '们', '那', '但',
    '可以', '这个', '那个', '什么', '怎么', '因为', '所以', '如果', '还是',
    '觉得', '可能', '知道', '里面', '出来', '一样', '时候', '然后', '不能',
    '不是', '就是', '比较', '这么', '那么', '一点', '有点', '一些', '很多',
    '还是', '不过', '之后', '以前', '只能', '只是', '的话', '啊', '吧',
    '哦', '嗯', '呢', '吗', '呀', '啦', '噢', '哟',
    # 影评常见无意义词
    '电影', '影片', '导演', '演员', '镜头', '剧情', '故事', '画面',
    '一部', '这部', '那个', '整个', '结局', '感觉', '真的', '特别',
    '非常', '比较', '有点', '还是', '其实', '可能', '很多', '一些',
    '拍摄', '表演', '角色', '人物', '里面', '出来', '看到', '看完',
    '喜欢', '觉得', '没有', '不是', '可以', '这个', '这么', '那么',
    '那种', '一种', '这种', '怎么', '什么',
])

# ==================== 6. 自定义议题增强词表 ====================
boost_words = set([
    '婚育', '母职', '父权', '厌女', '性侵', '性骚扰', '家暴', '重男轻女',
    '剩女', '催婚', '相亲', '生育', '子宫', '月经', '身体', '身材', '容貌',
    '穿衣自由', '独立女性', '全职妈妈', '家庭主妇', '职场歧视', '性同意',
    '荡妇', '羞辱', '压迫', '觉醒', '共情', '恐惧', '焦虑', '压抑',
    '暴力', '凝视', '物化', '规训', '羞耻', '歧视', '平等', '权利',
])

# ==================== 7. 分词 + 词性过滤：只留名词/动词 ====================
def extract_nouns_verbs(text):
    words = pseg.cut(text)
    allowed_flags = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn'}  # 名词与动词
    return ' '.join([w.word for w in words
                     if w.flag in allowed_flags
                     and w.word not in stopwords
                     and len(w.word) > 1])

corpus = negative_f[COM_TEXT].astype(str).apply(extract_nouns_verbs).tolist()

# ==================== 8. TF‑IDF 提取关键词 ====================
vectorizer = TfidfVectorizer(max_features=200, max_df=0.8, min_df=2)
try:
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_means = np.mean(tfidf_matrix, axis=0).A1
    word_scores = dict(zip(feature_names, tfidf_means))

    # 议题加权
    for w in word_scores:
        if w in boost_words:
            word_scores[w] *= 5.0   # 大幅提升女性议题词
        elif w in stopwords:
            word_scores[w] = 0.0

    # 归一化
    max_score = max(word_scores.values()) if word_scores else 1
    wordcloud_dict = {w: int(s / max_score * 1000) for w, s in word_scores.items() if s > 0}

except ValueError as e:
    print("TF-IDF出错，可能词汇太少，降级为简单词频")
    from collections import Counter
    wc = Counter()
    for doc in corpus:
        wc.update(doc.split())
    for w in wc:
        if w in boost_words:
            wc[w] *= 3
    wordcloud_dict = dict(wc)

# ==================== 9. 输出高频议题词表 ====================
sorted_words = sorted(wordcloud_dict.items(), key=lambda x: x[1], reverse=True)
print("\n===== 2020年电影女性议题负面评论 TOP30 议题关键词 =====")
for w, s in sorted_words[:30]:
    print(f"{w}\t{s}")

# ==================== 10. 生成词云 ====================
if wordcloud_dict:
    wc = WordCloud(
        font_path='simhei.ttf',   # 请确认字体路径
        width=1000,
        height=700,
        background_color='white',
        colormap='Reds',
        max_words=150
    ).generate_from_frequencies(wordcloud_dict)

    plt.figure(figsize=(12, 9))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('2020年电影女性议题负面评论的社会议题词云', fontsize=18)
    plt.tight_layout()
    plt.savefig('2020_movie_female_negative_wordcloud.png', dpi=300)
    plt.show()
else:
    print("无足够词汇生成词云")