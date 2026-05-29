import pandas as pd

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

# ==================== 2. 合并 + 锁定2024年 ====================
df = comments.merge(meta[[META_NAME, META_YEAR]], on=COM_NAME, how='left')
df[META_YEAR] = pd.to_numeric(df[META_YEAR], errors='coerce')
df = df.dropna(subset=[META_YEAR])
df[META_YEAR] = df[META_YEAR].astype(int)

df_2024 = df[df[META_YEAR] == 2024].copy()
print(f"2024年电影评论总数：{len(df_2024)}")

# ==================== 3. 确定情感数值列 ====================
if 'sentiment_score' in df_2024.columns:
    sent_col = 'sentiment_score'
    print("使用 sentiment_score")
else:
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_2024['_sent'] = df_2024[COM_LABEL].map(label_map)
    sent_col = '_sent'
    print("已映射 sentiment_label -> 数值")

# ==================== 4. 标记女性议题评论 ====================
female_keywords = [
    '她', '女人', '女性', '妇女', '女儿', '母亲', '妈妈', '妻子', '老婆',
    '婚姻', '结婚', '离婚', '生育', '生子', '子宫', '月经',
    '性别', '女性主义', '女权', '女拳', '直男癌', '厌女', '父权',
    '身材', '容貌', '穿衣自由', '化妆', '独立女性',
    '家庭主妇', '全职妈妈', '母职', '催婚', '相亲', '剩女'
]
df_2024['female_related'] = df_2024[COM_TEXT].apply(
    lambda x: any(kw in str(x) for kw in female_keywords)
)

# ==================== 5. 按电影分组统计 ====================
movie_stats = df_2024.groupby(COM_NAME).agg(
    评论总数=(COM_NAME, 'count'),
    情感均值=(sent_col, 'mean'),
    女性议题评论数=('female_related', 'sum')
).reset_index()

movie_stats['女性议题占比'] = movie_stats['女性议题评论数'] / movie_stats['评论总数']
movie_stats = movie_stats.sort_values('评论总数', ascending=False)

# ==================== 6. 输出峰值榜单 ====================
print("\n===== 2024年电影评论峰值榜（按评论数降序）=====")
print(movie_stats[['电影名', '评论总数', '情感均值', '女性议题占比']].to_string(index=False))

# 可选：保存为CSV
movie_stats.to_csv('2024_peak_movies.csv', index=False, encoding='utf-8-sig')
print("\n已保存详细数据至 2024_peak_movies.csv")