import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文字体设置（根据你的系统微调）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 文件路径 ====================
META_FILE = 'doulist_feminism_movies.csv'                 # 电影元数据
COMMENTS_FILE = 'douban_film_comments_with_sentiment.csv' # 影评情感数据

# ==================== 1. 读取数据 ====================
meta = pd.read_csv(META_FILE, encoding='utf-8')
comments = pd.read_csv(COMMENTS_FILE, encoding='utf-8')

# 确认列名（你的元数据：电影名、上映年份、评分；评论：电影名、评论内容、sentiment_label）
META_NAME = '电影名'
META_YEAR = '上映年份'
COMMENT_NAME = '电影名'
COMMENT_TEXT = '评论内容'
COMMENT_LABEL = 'sentiment_label'

# ==================== 2. 合并数据 ====================
df = comments.merge(meta[[META_NAME, META_YEAR]], on=COMMENT_NAME, how='left')
df[META_YEAR] = pd.to_numeric(df[META_YEAR], errors='coerce')
df = df.dropna(subset=[META_YEAR])
df[META_YEAR] = df[META_YEAR].astype(int)

# 检查是否有情感分数列，如果没有则通过标签映射
if 'sentiment_score' in df.columns:
    SENT_COL = 'sentiment_score'
    print("使用 sentiment_score 列")
else:
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['_sent'] = df[COMMENT_LABEL].map(label_map)
    SENT_COL = '_sent'
    print("已映射 sentiment_label -> 数值（1/0/-1）")

# ==================== 3. 定义女性议题关键词 ====================
female_keywords = [
    '她', '女人', '女性', '妇女', '女儿', '母亲', '妈妈', '妻子', '老婆',
    '婚姻', '结婚', '离婚', '生育', '生子', '子宫', '月经',
    '性别', '女性主义', '女权', '女拳', '直男癌', '厌女', '父权',
    '身材', '容貌', '穿衣自由', '化妆', '独立女性',
    '家庭主妇', '全职妈妈', '母职', '催婚', '相亲', '剩女'
]

df['female_related'] = df[COMMENT_TEXT].apply(
    lambda x: any(kw in str(x) for kw in female_keywords)
)
print(f"女性议题评论占比：{df['female_related'].mean():.2%}")

# ==================== 4. 逐年统计 ====================
all_year = df.groupby(META_YEAR).agg(
    sent_all=(SENT_COL, 'mean'),
    count_all=(SENT_COL, 'count')
).reset_index()

female_year = df[df['female_related']].groupby(META_YEAR).agg(
    sent_female=(SENT_COL, 'mean'),
    count_female=(SENT_COL, 'count')
).reset_index()

yearly = all_year.merge(female_year, on=META_YEAR, how='left')
yearly['sent_female'] = yearly['sent_female'].where(yearly['count_female'] > 0, pd.NA)

# 剔除样本量过少的年份
yearly = yearly[yearly['count_all'] >= 10]

# ==================== 5. 绘图 ====================
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(yearly[META_YEAR], yearly['sent_all'], marker='o', linewidth=2,
         color='steelblue', label='全部影评情感均值')
ax1.plot(yearly[META_YEAR], yearly['sent_female'], marker='s', linewidth=2,
         color='crimson', label='女性议题影评情感均值')

ax1.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

ax1.set_xlabel('上映年份', fontsize=13)
ax1.set_ylabel('情感均值', fontsize=13)
ax1.grid(alpha=0.3)

# 右侧柱状图
ax2 = ax1.twinx()
ax2.bar(yearly[META_YEAR], yearly['count_female'], alpha=0.25,
        color='crimson', label='女性议题评论数量')
ax2.set_ylabel('女性议题评论数量', fontsize=13, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.title('豆瓣影评情感变迁：女性议题 vs 整体', fontsize=17, pad=20)
plt.tight_layout()
plt.savefig('female_movies_sentiment_trend.png', dpi=300, bbox_inches='tight')
plt.show()