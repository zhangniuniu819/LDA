import os, re, jieba
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("已设置 Hugging Face 镜像源")
import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP

# ---------------------------- 环境设置 ----------------------------

EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# ================== 自定义词典（保证女性主义词汇不被切碎） ==================
feminist_dict = [
    "女性主义", "女权主义", "父权制", "男性凝视", "性别平等", "性别歧视",
    "厌女", "me too", "主体性", "赋权", "第二性", "波伏娃", "大女主",
    "独立女性", "女性意识", "姐妹情谊", "女性互助", "性骚扰", "家暴",
    "物化女性", "身体自主", "身体书写", "性解放", "月经", "子宫",
    # 书籍特有术语
    "女性写作", "女性文学", "母职", "母女关系", "性别研究", "性别政治",
    "écriture féminine", "women's writing", "female author"
]
for w in feminist_dict:
    jieba.add_word(w)

# ================== 停用词（在电影脚本基础上微增书籍高频词） ==================
STOP_WORDS = [
    '的','了','在','是','我','有','和','就','不','人','都','一','一个',
    '上','也','很','到','说','要','去','你','会','着','没有','看','好',
    '自己','这','那','但','可以','这个','那个','什么','怎么','因为','所以',
    '如果','还是','觉得','可能','知道','出来','一样','时候','然后','不能',
    '不是','就是','比较','这么','那么','一点','有点','一些','很多','不过',
    '之后','以前','只能','只是','的话','啊','吧','哦','嗯','呢','吗','呀','啦',
    '本书','作者','读者','阅读','故事','小说','作品','书中','感觉','真的',
    '美国','中国','日本','韩国','英国','法国','德国','国家','世界','社会',
    '生活','时间','地方','事情','东西','情况','结果','这样','那里','哪里',
    '等等','之类','能够','应该','必须','需要','也许','大概','现在','过去',
    '未来','今天','明天','昨天','今年','去年','两个','三个','她','他','它','们',
    '文字','看过','看完','一部','特别','完全','其实','有点','算是','部分','整体',
    '内容','情节','人物','描写','看到','感受','理解','方式','电影','影片','导演',
    '演员','镜头','剧情','画面','拍摄','表演','角色','整个','结局','非常',
    '还是','我们','他们','她们','但是','最后','真是','这部','不错','不要',
    '问题','后面','感动','需要','片子','纪录片','演技','叙事','女主','喜欢',
    '前面','还有','时候','事情','方面','状态','第一','第二','第三','首先','其次',
    '总之','总的来说','而且','然而','因此','所以','不过','其实','当然','之类',
    '例如','比如','以及','及其','能够','必须','可能','也许','大概','大约',
    '几乎','差不多','基本上','大部分','主要','重要','现在','过去','未来',
    '今天','明天','昨天','今年','去年','世界','全球','国际','国内','国外',
    '地方','地区','城市','读完','经历','起来','豆瓣','才能','作为','了解',
    '不同','不会','读到','啊啊啊','文学','文笔','结构','章节','出版','出版社',
    '译本','翻译','封面','装帧','评分','五星','四星','三星','推荐','畅销',
    '经典','名著','认为','以为','感到','好看','情感','文化','两个','一定',
    '成为','有些','一起','观众','台词','熊猫','动画','黎巴嫩','奥斯卡',
    '特效','配乐','无','真','太','很','好','坏','想','每','即','与','对',
    '而','可','曾','已','该','为','者','之','于','此','乃','乎',
    '曰','日','月','时','分','秒','今','昨','明','年','岁','次','个','些','种',
    '条','只','把','被','让','向','从','以','到','在','更','比','吗','呢','吧','啊','哦',
    # 书籍评论额外高频词
    '电子书','kindle','纸质','书评','书单','购书','图书馆','借阅','重读',
    '速度','流畅','晦涩','易懂','难读','入门','理论','学术','研究','资料',
    '注释','参考文献','索引','附录','前言','后记','序言','版本','印刷',
    '排版','字体','纸张','插图','图文','彩色','黑白','精装','平装',
    '作者本人','译者','翻译腔','信达雅','原文','原著','原版','中译本',
    '英文版','中文版','外国文学','中国文学','当代文学','古典文学',
    '诗歌','散文','戏剧','童话','科幻','悬疑','推理','言情','耽美',
    '网络小说','严肃文学','通俗小说','类型小说'
]

# ================== 同义词映射 ==================
SYNONYM_MAP = {
    "女权主义": "女性主义",
    "女权": "女性主义",
    "metoo": "me too",
    "domestic violence": "家暴",
    "sexual harassment": "性骚扰",
    "male gaze": "男性凝视",
    "patriarchy": "父权",
    "sisterhood": "姐妹情谊",
    "female protagonist": "女主角",
    "gender equality": "性别平等",
    "empowerment": "赋权",
    "objectification": "物化",
    "women's writing": "女性写作",
    "écriture féminine": "身体写作",
    "female author": "女作家"
}

# ---------------------------- 分词器 ----------------------------
def tokenize_zh(text):
    for old, new in SYNONYM_MAP.items():
        text = text.replace(old, new)
    tokens = [tok for tok in jieba.cut(text) if len(tok.strip()) > 1]
    return tokens

# ---------------------------- 加载数据 ----------------------------
print("加载书籍评论...")
df = pd.read_csv('douban_books_comments.csv')
docs = df['评论内容'].dropna().astype(str).tolist()
print(f"书籍评论数：{len(docs)}")

# ---------------------------- 书籍专属种子主题 ----------------------------
feminist_seed_topics_books = [
    # 理论、思想、运动
    ["女性主义","feminism","父权","男性凝视","性别平等","性别歧视",
     "厌女","misogyny","me too","主体性","赋权","第二性","波伏娃","性别研究"],
    # 女性写作与文学批评
    ["女性写作","女性文学","身体写作","女性叙事","女作家","女性视角",
     "女性主义文学","women's writing","écriture féminine"],
    # 女性成长、身份、家庭关系
    ["女性成长","身份认同","独立女性","女性困境","母亲身份","母职",
     "母女关系","女性友谊","姐妹情谊","女性力量","女性意识觉醒"],
    # 性别暴力、不平等
    ["性别歧视","性骚扰","家暴","物化女性","性别刻板印象","厌女",
     "重男轻女","性别不平等","性暴力"],
    # 身体、性、生育
    ["身体自主","生育权","月经","性","sexuality","堕胎","abortion",
     "身体书写","性解放","子宫"],
    # 女性主义实践与社群
    ["女权运动","me too","女性互助","女性社群","feminist activism",
     "姐妹情谊","女性空间"]
]

# ---------------------------- 模型构建 ----------------------------
sentence_model = SentenceTransformer(EMBEDDING_MODEL)

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

vectorizer_model = CountVectorizer(
    tokenizer=tokenize_zh,
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=5,
    stop_words=STOP_WORDS
)

representation_model = {
    "Main": KeyBERTInspired(),
    "Diverse": MaximalMarginalRelevance(diversity=0.3)
}

topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    seed_topic_list=feminist_seed_topics_books,
    nr_topics=30,
    min_topic_size=20,
    top_n_words=15,
    calculate_probabilities=True,
    verbose=True
)

# ---------------------------- 训练 ----------------------------
print("训练书籍模型...")
topics, probs = topic_model.fit_transform(docs)

# 保存模型与结果
topic_model.save("bertopic_book_optimized")
topic_info = topic_model.get_topic_info()
topic_info.to_csv("book_topic_info_optimized.csv", index=False, encoding='utf-8-sig')

# ---------------------------- HTML 可视化 ----------------------------
print("生成书籍可视化...")
topic_model.visualize_barchart(top_n_topics=20, n_words=10).write_html("book_barchart.html")
topic_model.visualize_topics().write_html("book_topics_map.html")
topic_model.visualize_hierarchy(top_n_topics=30).write_html("book_hierarchy.html")
topic_model.visualize_heatmap().write_html("book_heatmap.html")

if 1 in topic_info.Topic.values:
    topic_model.visualize_term_rank(topics=[1]).write_html("book_term_rank_topic1.html")
print("所有书籍主题图表已生成。")