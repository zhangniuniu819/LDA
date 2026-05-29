import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import warnings
warnings.filterwarnings('ignore')

# ==================== 0. 参数设置（按需修改） ====================
LEXICON_PATH = '情感词汇本体.xlsx'     # 情感词典文件路径
BOOKS_PATH   = 'douban_books_comments.csv'   # 书籍评论
FILMS_PATH   = 'doulist_film_comments.csv'   # 电影评论
TALKS_PATHS = [
    'talk_show_sentences.csv',      # 第一个文件
    'talk_show_sentences(2).csv',     # 第二个文件
    'talk_show_sentences(1).csv',     # 第三个文件
    'talk_show_sentences（4）.csv'      # 第四个文件
]

# ==================== 读取并合并三类数据 ====================
def read_and_label(filepath, media_label):
    """读取单个 CSV，自动查找文本列，并添加媒介标签"""
    df = pd.read_csv(filepath)
    text_col_candidates = ['Text']
    text_col = None
    for cand in text_col_candidates:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        text_col = df.columns[0]
        print(f'⚠️ 文件 {filepath} 未找到文本列，默认使用第一列: {text_col}')
    df = df.rename(columns={text_col: 'text'})
    df = df[['text']]
    df['media'] = media_label
    return df
TALKS_PATH   = 'talk/talk_show_sentences.csv'  # 脱口秀分句
OUTPUT_FIG   = 'cross_media_emotion_violin.png'

# ==================== 1. 加载情感词典并构建加权查找表 ====================
print('[1/5] 正在加载情感词典...')
df_lex = pd.read_excel(LEXICON_PATH)

# ----- 自动识别列名（常见变体） -----
col_map = {
    '词语': ['词语', 'word', '词', 'Word'],
    '情感分类': ['情感分类', 'emotion', '情绪', '情感类别', 'Emotion'],
    '强度': ['强度', 'intensity', 'Intensity', '情感强度']
}

rename_dict = {}
for std_name, possible_names in col_map.items():
    for col in df_lex.columns:
        if col.strip() in possible_names:
            rename_dict[col] = std_name
            break

if len(rename_dict) < 3:
    # 若自动识别失败，则提示用户手动修改列名
    print('⚠️ 未能自动识别列名，当前列名：', list(df_lex.columns))
    print('请将必要列名改为：词语、情感分类、强度 后重新运行')
    # 这里也可以让用户输入对应的列名，但为简洁直接退出
    raise KeyError('列名映射失败，请检查词典文件结构。')

df_lex.rename(columns=rename_dict, inplace=True)

# 删除有缺失的关键字段
df_lex = df_lex.dropna(subset=['词语', '情感分类', '强度'])
print(f'词典加载完成，有效词条数: {len(df_lex)}')

# 构建加权字典：{词: [(情绪, 强度), ...]}
# ----- 自动识别列名（保持原样） -----
col_map = {
    '词语': ['词语', 'word', '词', 'Word'],
    '情感分类': ['情感分类', 'emotion', '情绪', '情感类别', 'Emotion'],
    '强度': ['强度', 'intensity', 'Intensity', '情感强度']
}
rename_dict = {}
for std_name, possible_names in col_map.items():
    for col in df_lex.columns:
        if col.strip() in possible_names:
            rename_dict[col] = std_name
            break
if len(rename_dict) < 3:
    print('列名映射失败，当前列名：', list(df_lex.columns))
    raise KeyError('请检查词典文件列名。')
df_lex.rename(columns=rename_dict, inplace=True)

df_lex = df_lex.dropna(subset=['词语', '情感分类', '强度'])
print(f'词典加载完成，有效词条数: {len(df_lex)}')

# ========== 新增：英文代码 → 中文映射表 ==========
emotion_code_map = {
    'PA': '乐', '乐': '乐',
    'PD': '好', '好': '好',
    'NA': '怒', '怒': '怒',
    'NB': '哀', '哀': '哀',
    'NC': '惧', '惧': '惧',
    'ND': '恶', '恶': '恶',
    'NE': '惊', '惊': '惊',
    # 如果词典中还有其他表示（如 PE），请根据实际情况补充
}
# 可先看一眼有哪些情感编码（运行一次打印后，再注释掉）
# print(df_lex['情感分类'].value_counts())

# ========== 构建加权字典（只保留七大情绪） ==========
word_emotion_weighted = {}
for _, row in df_lex.iterrows():
    word = str(row['词语']).strip()
    emotion_raw = row['情感分类'].strip() if isinstance(row['情感分类'], str) else row['情感分类']
    emotion = emotion_code_map.get(emotion_raw, None)   # 映射为中文，没有则忽略
    if emotion is None:
        continue
    intensity = row['强度']
    if word not in word_emotion_weighted:
        word_emotion_weighted[word] = []
    word_emotion_weighted[word].append((emotion, intensity))

print(f'映射后有效情感词数: {len(word_emotion_weighted)}')

# ==================== 2. 向 jieba 添加词典词，防止误切 ====================
print('[2/5] 正在配置 jieba 分词...')
for w in word_emotion_weighted:
    jieba.add_word(w)   # 保证情感词汇能被整体识别
print(f'已添加 {len(word_emotion_weighted)} 个自定义词到 jieba')

# ==================== 3. 定义加权情感向量计算函数 ====================
def calc_emotion_vector(text):
    """
    返回长度为7的情感向量 [乐,好,怒,哀,惧,恶,惊]，
    每个维度 = 该情绪的累积强度 / 所有情绪总强度
    """
    if not isinstance(text, str):
        return [0.0] * 7

    words = jieba.lcut(text)
    emo_intensities = {e: 0.0 for e in ['乐', '好', '怒', '哀', '惧', '恶', '惊']}
    total_intensity = 0.0

    for w in words:
        if w in word_emotion_weighted:
            for emotion, intens in word_emotion_weighted[w]:
                emo_intensities[emotion] += intens
                total_intensity += intens

    if total_intensity == 0:
        return [0.0] * 7

    emo_order = ['乐', '好', '怒', '哀', '惧', '恶', '惊']
    vec = [emo_intensities[emo] / total_intensity for emo in emo_order]
    return vec

# ==================== 4. 读取并合并三类数据 ====================
print('[3/5] 正在读取并合并数据...')

def read_and_label(filepath, media_label):
    """读取 CSV，自动查找文本列，并添加媒介标签"""
    df = pd.read_csv(filepath)
    # 常见文本列名
    text_col_candidates = ['评论内容', '句子', 'Text', 'text', '内容', 'comment']
    text_col = None
    for cand in text_col_candidates:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        # 找不到就用第一列（假设第一列是文本）
        text_col = df.columns[0]
        print(f'⚠️ 文件 {filepath} 未找到文本列，默认使用第一列: {text_col}')
    df = df.rename(columns={text_col: 'text'})
    df = df[['text']]   # 只保留文本列
    df['media'] = media_label
    return df

df_books = read_and_label(BOOKS_PATH, 'book')
df_films = read_and_label(FILMS_PATH, 'film')

# 读取多个脱口秀文件
df_talks_list = []
for path in TALKS_PATHS:
    print(f'正在读取脱口秀文件: {path}')
    df_tmp = read_and_label(path, 'talk')
    df_talks_list.append(df_tmp)
df_talks = pd.concat(df_talks_list, ignore_index=True)

df_all = pd.concat([df_books, df_films, df_talks], ignore_index=True)
print(f'合并后总文本数：{len(df_all)}')
print('各类媒介样本数：')
print(df_all['media'].value_counts())


# ==================== 5. 计算情感向量 ====================
print('[4/5] 正在计算情感向量（可能需要一点时间）...')
emo_labels = ['乐', '好', '怒', '哀', '惧', '恶', '惊']

# 使用 apply 批量计算
emo_arrays = df_all['text'].apply(calc_emotion_vector)
emo_df = pd.DataFrame(emo_arrays.tolist(), columns=emo_labels)

# 最终数据表：media + 7维情感占比
df_result = pd.concat([df_all[['media']], emo_df], axis=1)

# 简单统计：各媒介平均情感向量
print('\n各媒介平均情感向量（加权强度占比）：')
print(df_result.groupby('media')[emo_labels].mean().round(3))

# 标记哪些文本未能匹配到情感词（全是0）
hit_mask = (df_result[emo_labels].sum(axis=1) == 0)
print(f'未匹配到任何情感词的文本数: {hit_mask.sum()} (占{hit_mask.mean():.2%})')
# 可选：保存中间结果
df_result.to_csv('emotion_vectors.csv', index=False, encoding='utf-8-sig')
print('情感向量已保存到 emotion_vectors.csv')

# ==================== 6. 绘制小提琴图 ====================
print('[5/5] 绘制小提琴图...')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, emo in enumerate(emo_labels):
    ax = axes[i]
    # 为了可视化更清晰，可过滤掉全零样本，但这里保留全部对比
    sns.violinplot(x='media', y=emo, data=df_result,
                   palette='Set2', ax=ax, inner='quartile',
                   cut=0)  # cut=0 限制小提琴范围在数据范围内
    ax.set_title(f'{emo} 情感强度分布', fontsize=13)
    ax.set_xlabel('媒介')
    ax.set_ylabel('加权强度占比')

# 删除多余子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('书籍 vs 电影 vs 脱口秀 情感轮廓对比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
plt.show()
print(f'✅ 图片已保存为 {OUTPUT_FIG}')
