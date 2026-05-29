import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("已设置 Hugging Face 镜像源")
import re
import glob
import pandas as pd
import numpy as np
import jieba
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP

# ---------------------------- 环境设置 ----------------------------


# ---------------------------- 白名单（自定义词典，保证关键术语不被切碎）------------------------
WHITELIST = [
    '女性', '男性', '性别', '女人', '男人', '女孩', '男孩', '女生', '男生',
    '母亲', '妈妈', '父亲', '爸爸', '女儿', '儿子', '妻子', '丈夫', '老婆', '老公',
    '孩子', '儿童', '婴儿', '孕妇', '产妇',
    '女性主义', '女权', '女权主义', '性别平等', '性别歧视', '平权', '父权制', '父权',
    '厌女', '厌女症', '男性凝视', '凝视', '物化', '刻板印象', '刻板',
    '身体', '身材', '容貌', '外貌', '颜值', '穿衣自由', '性骚扰', '性侵', '强奸',
    '性同意', '生育', '堕胎', '堕胎权', '代孕', '母职', '母职惩罚', '育儿', '家务',
    '身体自主权', '容貌焦虑', '身材焦虑', '哺乳', '经期', '月经',
    '婚姻', '结婚', '离婚', '婚恋', '嫁娶', '家庭', '家务劳动', '丧偶式育儿',
    '全职妈妈', '家庭主妇', '职场妈妈', '单亲妈妈',
    '职场', '工作', '职业', '就业', '歧视', '玻璃天花板', '经济独立', '独立女性',
    '职场歧视', '同工同酬', '产假', '哺乳假',
    '暴力', '家暴', '家庭暴力', '性暴力', '压迫', '反抗', '觉醒', '挣扎', '自由',
    '权利', '权益', '平等', '独立', '自主', '选择', '声音', '发言权',
    '同性恋', '女同性恋', '拉拉', '同志', 'LGBT', '跨性别', '酷儿', '出柜',
    '芭比', '肯', '母权', '女巫', '猎巫', 'metoo', '反性骚扰',
    '晋升', '升职', '加薪', '工资', '薪酬', '收入', '失业', '待业',
    '领导', '管理', '高管', '决策', '开会', '出差', '加班', '996',
    '面试', '简历', '招聘', '性别比', '男女比例', '工科', '理科',
    '创业', '老板', '上司', '同事', '团队', '合作', '职场霸凌',
    '彩礼', '嫁妆', '婚房', '户口', '生育率', '二胎', '三胎',
    '催婚', '相亲', '逼婚', '单身', '剩女', '大龄', '不婚',
    '丁克', '分居', '出轨', '婚外情', '小三', '离婚冷静期',
    '抚养权', '赡养', '养老', '婆媳', '妯娌', '大男子主义',
    '贤惠', '温顺', '听话', '懂事', '牺牲', '奉献', '付出', '包容',
    '减肥', '瘦身', '苗条', '体重', '饮食', '节食', '健身',
    '化妆', '素颜', '口红', '穿搭', '医美', '整容', '隆胸',
    '美白', '防晒', '脱毛', '姨妈', '痛经', '更年期', '绝经',
    '妇科', '乳腺', '子宫', '卵巢', 'HPV', '宫颈', '避孕药',
    '卫生巾', '月经杯', '月嫂', '催乳', '产后抑郁',
    '重男轻女', '传宗接代', '续香火', '养儿防老', '偏心',
    '性别刻板', '标签', '污名', '荡妇羞辱', '受害者有罪论',
    '男权', '女尊', '平权', '激进', '保守', '刻板印象',
    '偏见', '歧视', '优待', '逆歧视', '反歧视', '权益', '维权',
    '上学', '辍学', '升学', '高考', '专业', '文科', '理科',
    '学历', '硕士', '博士', '女博士', '学霸', '学渣',
    '早恋', '青春期', '叛逆', '家庭教育', '子女教育',
    '强奸犯', '猥亵', '偷拍', '性侵害', '暴力', '殴打', '虐待',
    '冷暴力', '精神控制', 'PUA', '恐吓', '跟踪', '尾随',
    '报警', '诉讼', '法律', '庇护所', '保护令',
    '报道', '新闻', '热搜', '舆论', '公知', '大V', '媒体',
    '言论', '禁言', '删帖', '封号', '举报', '意识形态',
    '女权运动', '游行', '集会', '请愿', '联名', '罢工',
    '志愿者', '公益', 'NGO', '基金会', '奖学金', '互助',
    '娘炮', '伪娘', '女汉子', '女装', '男装', '中性',
    '柔弱', '坚强', '勇敢', '脆弱', '刚强', '温柔',
    '霸道', '任性', '撒娇', '卖萌', '绿茶', '白莲花',
    '小说', '影视', '剧集', '角色', '女主角', '男主角',
    '剧情', '编剧', '导演', '作家', '书评', '影评'
]

# 去重并添加到jieba词典
WHITELIST = list(set(WHITELIST))
for word in WHITELIST:
    jieba.add_word(word)


# ---------------------------- 停用词加载（从文件读取）------------------------
def load_stopwords_from_file(filepath="停用词.txt", default_stopwords=None):
    """
    从文本文件读取停用词（每行一个词）。
    如果文件不存在，则使用默认列表。

    Args:
        filepath: 停用词文件路径
        default_stopwords: 默认停用词列表

    Returns:
        去重后的停用词列表
    """
    if default_stopwords is None:
        default_stopwords = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"成功从 {filepath} 加载 {len(words)} 个停用词")
        return list(set(words))  # 去重
    except FileNotFoundError:
        print(f"警告：{filepath} 未找到，使用默认停用词列表")
        return list(set(default_stopwords))
    except Exception as e:
        print(f"读取停用词文件出错：{e}，使用默认列表")
        return list(set(default_stopwords))




# 加载停用词
STOP_WORDS = load_stopwords_from_file("停用词.txt")
print(f"最终停用词表大小：{len(STOP_WORDS)}")

# ---------------------------- 同义词映射（合并近义词）------------------------
SYNONYM_MAP = {
    "女权主义": "女性主义",
    "女权": "女性主义",
    "父权制": "父权",
    "厌女症": "厌女",
    "男性凝视": "凝视",
    "刻板印象": "刻板",
    "metoo": "MeToo",
    "家务劳动": "家务",
    "玻璃天花板": "职场歧视",
    "LGBT": "性少数",
    "跨性别": "性少数",
    "酷儿": "性少数",
    "女同性恋": "拉拉",
    "家暴": "家庭暴力",
    "性侵": "性侵害",
    "精神控制": "PUA",
    "独立女性": "女性独立",
    "穿衣自由": "着装自由",
    "身体自主权": "身体自主",
    "容貌焦虑": "外貌焦虑",
    "身材焦虑": "体型焦虑",
    "育儿假": "育儿假期",
    "哺乳假": "产假",
    "离婚冷静期": "离婚制度",
    "传宗接代": "延续香火",
    "荡妇羞辱": "性污名",
    "受害者有罪论": "归咎受害者",
    "女权运动": "女性运动",
    "同工同酬": "薪资平等",
    "职场霸凌": "工作欺凌",
    "大男子主义": "男性中心",
    "服美役": "外貌强迫",
    "减肥": "瘦身",
    "整容": "医美",
    "痛经": "经期疼痛",
    "更年期": "围绝经期",
    "卫生巾": "月经用品",
    "月嫂": "产后护理",
    "产后抑郁": "产后忧郁",
    "重男轻女": "性别偏袒",
    "性别刻板": "性别定型",
    "平权": "平等权利",
    "女博士": "高学历女性",
    "剩女": "未婚女性",
    "娘炮": "阴柔男性",
    "女汉子": "阳刚女性"
}


# ---------------------------- 分词器（同义词替换 + jieba分词 + 停用词过滤）------------------------
def tokenize_zh(text):
    """
    中文分词函数：
    1. 同义词替换
    2. jieba分词
    3. 停用词过滤
    4. 长度过滤（保留>=2个字符）

    Args:
        text: 待分词的文本

    Returns:
        分词后的词列表
    """
    # 空值处理
    if pd.isna(text) or text.strip() == "":
        return []

    # 转为字符串并去除首尾空格
    text = str(text).strip()

    # 同义词替换
    for old, new in SYNONYM_MAP.items():
        text = text.replace(old, new)

    # 分词，只保留长度>=2且不在停用词表中的词
    tokens = []
    for tok in jieba.cut(text):
        clean_tok = tok.strip()
        if len(clean_tok) == 2 and clean_tok not in STOP_WORDS:
            tokens.append(clean_tok)

    return tokens


# ---------------------------- 加载数据 ----------------------------
print("正在加载脱口秀分句文件...")
file_patterns = [
    "talk_show_sentences(1).csv",
    "talk_show_sentences(2).csv",
    "talk_show_sentences.csv",
    'talk_show_sentences（4）.csv'
]

df_list = []
for pattern in file_patterns:
    try:
        # 尝试读取文件
        df_temp = pd.read_csv(pattern, encoding='utf-8-sig')

        # 检查必要的列
        if 'Text' not in df_temp.columns:
            print(f"警告：{pattern} 中没有找到 'Text' 列，跳过该文件。")
            continue

        # 只保留Text列并去重
        df_temp = df_temp[['Text']].copy()
        df_temp = df_temp.dropna(subset=['Text'])
        df_temp['Text'] = df_temp['Text'].astype(str).str.strip()
        df_temp = df_temp[df_temp['Text'] != ""]

        df_list.append(df_temp)
        print(f"成功加载 {pattern}，共 {len(df_temp)} 行有效数据")

    except FileNotFoundError:
        print(f"文件不存在：{pattern}，请检查路径。")
    except Exception as e:
        print(f"读取 {pattern} 时出错：{e}")

# 检查是否加载到数据
if not df_list:
    raise ValueError("没有成功加载任何数据文件，请确认文件存在且包含 'Text' 列。")

# 合并所有数据并去重
df = pd.concat(df_list, ignore_index=True)
df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
docs = df['Text'].tolist()

print(f"总计加载 {len(docs)} 条不重复的分句文本。")

# 预览前3条数据
for i in range(min(3, len(docs))):
    print(f"示例 {i + 1}: {docs[i][:100]}...")

# ---------------------------- BERTopic 模型组件 ----------------------------
print("初始化BERTopic模型组件...")

# 嵌入模型
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# UMAP降维模型
umap_model = UMAP(
    n_neighbors=10,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
    low_memory=False
)

# 向量化模型（使用自定义分词器）
vectorizer_model = CountVectorizer(
    tokenizer=tokenize_zh,
    ngram_range=(1, 2),  # 保留单字和双字词
    max_df=0.85,  # 过滤出现频率超过85%的词
    min_df=3,  # 只保留至少出现2次的词
    max_features=10000,  # 限制最大特征数
    lowercase=False  # 中文不需要小写
)

# 主题表示模型
representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "MMR": MaximalMarginalRelevance(diversity=0.3)
}

# 种子主题（脱口秀相关）
seed_topics = [
    ["婚姻", "结婚", "离婚", "婚恋", "催婚", "相亲", "单身", "剩女"],
    ["职场", "工作", "老板", "加班", "996", "晋升", "工资", "面试", "招聘", "职场歧视"],
    ["性别歧视", "厌女", "父权", "男性凝视", "物化", "刻板印象", "荡妇羞辱"],
    ["身体", "容貌", "减肥", "整容", "化妆", "穿衣自由", "服美役", "身材焦虑"],
    ["生育", "母亲", "妈妈", "母职", "育儿", "家务", "全职妈妈", "丧偶式育儿"],
    ["性骚扰", "性侵", "家暴", "冷暴力", "PUA", "精神控制", "恐吓"],
    ["月经", "卫生巾", "痛经", "更年期", "妇科", "子宫", "HPV"],
    ["女性主义", "女权", "平权", "MeToo", "独立女性", "女性互助", "姐妹情谊"],
    ["家庭", "父母", "孩子", "教育", "重男轻女", "彩礼", "嫁妆", "婆媳"],
    ["影视", "芭比", "肯", "角色", "女主角", "剧情", "导演"],
    ["脱口秀", "段子", "梗", "吐槽", "观众", "笑声", "舞台", "炸场"],
    ["女性主义", "性别平等", "女性权益", "职场歧视", "性别刻板印象", "婚姻自由", "生育自主权"],
    ["脱口秀", "女性脱口秀", "性别议题", "女性表达", "反性别歧视", "女性声音"],
    ["婚姻", "离婚", "彩礼", "女性困境", "婚姻压迫", "家庭分工"],
    ["职场", "女性职场", "玻璃天花板", "同工不同酬", "职场性骚扰", "性别歧视"],
    ["生育", "生育自由", "生育成本", "女性身体自主权", "母职惩罚"],
    ["性别暴力", "家庭暴力", "性暴力", "受害者保护", "反家暴"],
    ["女性成长", "女性觉醒", "独立女性", "自我价值", "女性力量"]
]

# 创建BERTopic模型
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    seed_topic_list=seed_topics,
    nr_topics=10,  # 目标主题数量
    min_topic_size=10,  # 最小主题文档数
    top_n_words=8,  # 每个主题返回的关键词数
    calculate_probabilities=True,  # 计算主题概率
    verbose=True,  # 详细输出
    language="chinese",  # 指定中文
    low_memory=False  # 关闭低内存模式以提高性能
)

# ---------------------------- 训练模型 ----------------------------
print("开始训练 BERTopic 模型...")
topics, probs = topic_model.fit_transform(docs)
print("模型训练完成。")
# ---------------------------- 保存结果 ----------------------------
print("保存模型和结果文件...")

# 保存模型
topic_model.save("bertopic_talk_show_model")
print("✅ 模型已保存为 bertopic_talk_show_model")

# 保存主题信息
topic_info = topic_model.get_topic_info()
topic_info.to_csv("talk_show_topic_info.csv", index=False, encoding='utf-8-sig')
print("✅ 主题信息已保存至 talk_show_topic_info.csv")

# 保存带主题标签的文档
df['topic'] = topics
df['probability'] = probs.max(axis=1) if probs is not None else 0.0
df.to_csv("talk_show_with_topics.csv", index=False, encoding='utf-8-sig')
print("✅ 带主题标签的文档已保存至 talk_show_with_topics.csv")

# ---------------------------- 可视化 ----------------------------
print("生成可视化图表...")

# 生成各类可视化图表
try:
    # 主题关键词条形图
    topic_model.visualize_barchart(top_n_topics=20, n_words=10).write_html("talk_show_barchart.html")

    # 主题分布图
    topic_model.visualize_topics().write_html("talk_show_topics_map.html")

    # 主题层次结构图
    topic_model.visualize_hierarchy(top_n_topics=30).write_html("talk_show_hierarchy.html")

    # 主题相关性热力图
    topic_model.visualize_heatmap().write_html("talk_show_heatmap.html")

    # 主题术语排名图（如果主题1存在）
    if 1 in topic_info['Topic'].values:
        topic_model.visualize_term_rank(topics=[1]).write_html("talk_show_term_rank_topic1.html")

    print("✅ 所有可视化图表已生成完成")

except Exception as e:
    print(f"⚠️ 生成可视化图表时出现错误：{e}")
    print("部分图表可能未生成，请检查相关依赖包是否安装完整")

# ---------------------------- 输出最终统计信息 ----------------------------
print("\n" + "=" * 50)
print("模型训练完成，最终统计信息：")
print(f"- 总文档数：{len(docs)}")
print(f"- 生成主题数：{len(topic_info) - 1} (不包含异常主题)")
print(f"- 异常主题文档数：{len(df[df['topic'] == -1])}")
print(f"- 平均主题概率：{df['probability'].mean():.4f}")
print("=" * 50)