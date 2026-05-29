import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据 ==========
meta_df = pd.read_csv('douban_feminism_books_metadata.csv', encoding='utf-8')
comments_df = pd.read_csv('douban_books_comments_with_sentiment.csv', encoding='utf-8')

print("元数据列名：", meta_df.columns.tolist())
print("评论数据列名：", comments_df.columns.tolist())

# 列名确认（按你提供的）
meta_book_col = '书名'
meta_year_col = '出版年'
com_book_col = '书名'
com_text_col = '评论内容'
com_sent_col = 'sentiment_label'
com_sent_score_col = None  # 如果有情感分数列，优先用分数计算均值

# 检查评论数据里是否有情感分数列（之前可能生成过 sentiment_score）
if 'sentiment_score' in comments_df.columns:
    com_sent_score_col = 'sentiment_score'
    print("检测到情感分数列 'sentiment_score'，将使用数值计算均值")
else:
    print("未检测到情感分数列，将使用标签列 'sentiment_label' 映射为数值")
    # 映射：positive=1, neutral=0, negative=-1
    sent_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    comments_df['_sent_numeric'] = comments_df[com_sent_col].map(sent_map)

# ========== 2. 合并数据 ==========
df = comments_df.merge(meta_df[[meta_book_col, meta_year_col]],
                       left_on=com_book_col, right_on=meta_book_col, how='left')
print(f"合并前评论数：{len(comments_df)}，合并后：{len(df)}")

# 清理出版年
df[meta_year_col] = pd.to_numeric(df[meta_year_col], errors='coerce')
df = df.dropna(subset=[meta_year_col])
df[meta_year_col] = df[meta_year_col].astype(int)
print(f"有效年份记录数：{len(df)}")

# ========== 3. 女性议题关键词标记 ==========
female_keywords = [
    '她', '女人', '女性', '妇女', '女儿', '母亲', '妈妈', '妻子', '老婆',
    '婚姻', '结婚', '离婚', '生育', '生子', '子宫', '月经',
    '性别', '女性主义', '女权', '女拳', '直男癌', '厌女', '父权',
    '身材', '容貌', '穿衣自由', '化妆', '独立女性',
    '家庭主妇', '全职妈妈', '母职', '催婚', '相亲', '剩女'
]

def is_female(text):
    text = str(text)
    return any(kw in text for kw in female_keywords)

df['female_related'] = df[com_text_col].apply(is_female)
print(f"女性议题评论占比：{df['female_related'].mean():.2%}")

# ========== 4. 确定用于计算均值的情感数值列 ==========
if com_sent_score_col:
    sent_val_col = com_sent_score_col
else:
    sent_val_col = '_sent_numeric'

# 全部评论逐年统计
all_year = df.groupby(meta_year_col).agg(
    mean_sent=(sent_val_col, 'mean'),
    count=('书名', 'count')
).reset_index()
all_year.columns = ['year', 'sent_all', 'count_all']

# 女性议题评论逐年统计
female_year = df[df['female_related']].groupby(meta_year_col).agg(
    mean_sent=(sent_val_col, 'mean'),
    count=('书名', 'count')
).reset_index()
female_year.columns = ['year', 'sent_female', 'count_female']

# 合并
yearly = all_year.merge(female_year, on='year', how='left')
yearly['sent_female'] = yearly['sent_female'].where(yearly['count_female'] > 0, pd.NA)

# 可选：过滤掉总数太少的年份（避免极端值）
yearly = yearly[yearly['count_all'] >= 10]

print("\n逐年统计表：")
print(yearly)

# ========== 5. 绘图 ==========
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(yearly['year'], yearly['sent_all'], marker='o', linewidth=2,
         color='steelblue', label='全部书评情感均值')
ax1.plot(yearly['year'], yearly['sent_female'], marker='s', linewidth=2,
         color='crimson', label='女性议题书评情感均值')

# 零线
ax1.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

ax1.set_xlabel('出版年份', fontsize=13)
ax1.set_ylabel('情感均值', fontsize=13)
ax1.grid(alpha=0.3)

# 右侧轴：评论数量柱状图
ax2 = ax1.twinx()
ax2.bar(yearly['year'], yearly['count_female'], alpha=0.25, color='crimson', label='女性议题评论数量')
ax2.set_ylabel('女性议题评论数量', fontsize=13, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

# 图例合并
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.title('豆瓣书评情感变迁：女性议题 vs 整体', fontsize=17, pad=20)
plt.tight_layout()
plt.savefig('female_books_sentiment_trend.png', dpi=300, bbox_inches='tight')
plt.show()