import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from scipy import stats

# ========== 1. 读取数据 ==========
df = pd.read_csv('电影对比.csv')
df.columns = [c.strip() for c in df.columns]

# ---------- 缺失值检查 ----------
total = len(df)
missing_nonfem = df['同导演的非女性主义代表作'].isna() | df['评分2'].isna()
has_pair = ~missing_nonfem  # 有完整配对的行

print(f"总电影数: {total}")
print(f"有非女性主义配对作品的: {has_pair.sum()}")
print(f"缺失非女性主义配对作品的: {missing_nonfem.sum()}")
if missing_nonfem.sum() > 0:
    print("以下电影缺失非女性主义代表作/评分2，将在配对分析中跳过：")
    print(df.loc[missing_nonfem, ['电影名', '同导演的非女性主义代表作', '评分2']])

# ========== 2. 准备数据 ==========
# 女性主义电影数据（全部）
fem_name = df['电影名']
fem_rating = df['评分']
fem_year = df['上映年份']

# 只取有非女性主义评分的行用于配对
paired_df = df[has_pair].copy()
nonfem_name = paired_df['同导演的非女性主义代表作']
nonfem_rating = paired_df['评分2']

# 创建长格式对比数据：女性主义电影全部保留，非女性主义只保留有评分的
compare_list = []
# 全部女性主义电影
compare_list.append(pd.DataFrame({
    '电影': fem_name,
    '评分': fem_rating,
    '类型': '女性主义',
    '年份': fem_year
}))
# 只添加有评分的非女性主义电影
if len(paired_df) > 0:
    compare_list.append(pd.DataFrame({
        '电影': nonfem_name,
        '评分': nonfem_rating,
        '类型': '非女性主义',
        '年份': paired_df['上映年份']  # 使用对应女性主义电影的年份
    }))

compare_df = pd.concat(compare_list, ignore_index=True)
compare_df = compare_df.dropna(subset=['评分'])
print("\n用于整体对比的数据量：")
print(compare_df['类型'].value_counts())

# ========== 3. 全局设置 ==========
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 4. 图1：按年份柱状图+两条折线图 ==========
yearly_fem = df.groupby('上映年份').agg(
    女性主义作品数量=('电影名', 'count'),
    女性主义平均评分=('评分', 'mean')
).reset_index()

if len(paired_df) > 0:
    yearly_nonfem = paired_df.groupby('上映年份')['评分2'].mean().reset_index()
    yearly_nonfem.rename(columns={'评分2': '非女性主义平均评分'}, inplace=True)
else:
    yearly_nonfem = pd.DataFrame(columns=['上映年份', '非女性主义平均评分'])

yearly_summary = yearly_fem.merge(yearly_nonfem, on='上映年份', how='left')
yearly_summary = yearly_summary.sort_values('上映年份')

fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.bar(yearly_summary['上映年份'].astype(str), yearly_summary['女性主义作品数量'],
        color='#FFB6C1', alpha=0.8, label='女性主义作品数量')
ax1.set_xlabel('上映年份')
ax1.set_ylabel('作品数量', color='#FF69B4')
ax1.tick_params(axis='y', labelcolor='#FF69B4')
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax2 = ax1.twinx()
ax2.plot(yearly_summary['上映年份'].astype(str), yearly_summary['女性主义平均评分'],
         color='#DC143C', marker='o', linewidth=2, label='女性主义平均评分')
if not yearly_nonfem.empty:
    ax2.plot(yearly_summary['上映年份'].astype(str), yearly_summary['非女性主义平均评分'],
             color='#4169E1', marker='s', linestyle='--', linewidth=2, label='非女性主义平均评分')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('女性主义电影作品数量与两类作品平均评分随年份变化趋势')
fig.tight_layout()
plt.savefig('趋势对比图.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 图2：配对差异图（仅当有配对数据时） ==========
if len(paired_df) > 0:
    pair_df = paired_df.dropna(subset=['评分', '评分2']).copy()
    pair_df['评分差'] = pair_df['评分'] - pair_df['评分2']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # 连线图
    ax = axes[0]
    for i, row in pair_df.iterrows():
        ax.plot([0, 1], [row['评分2'], row['评分']], 'o-', color='gray', alpha=0.6, markersize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['非女性主义', '女性主义'])
    ax.set_ylabel('评分')
    ax.set_title('同导演作品评分对比（每条线代表一位导演）')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 差值分布
    ax = axes[1]
    sns.histplot(pair_df['评分差'], bins=12, kde=True, color='#9370DB', ax=ax)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='无差异线')
    ax.set_xlabel('评分差 (女性主义 - 非女性主义)')
    ax.set_title('配对评分差的分布')
    ax.legend()

    fig.suptitle('同导演作品评分配对比较（仅包含有配对数据的导演）', fontsize=16)
    plt.tight_layout()
    plt.savefig('配对比较图.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\n⚠️ 没有非女性主义配对数据，跳过配对比较图。")

# ========== 6. 图3：整体分布对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(x='类型', y='评分', data=compare_df, palette=['#FFB6C1', '#87CEEB'], ax=axes[0])
axes[0].set_title('评分分布箱线图')
axes[0].set_xlabel('')
axes[0].set_ylabel('评分')

sns.violinplot(x='类型', y='评分', data=compare_df, palette=['#FFB6C1', '#87CEEB'], ax=axes[1])
axes[1].set_title('评分分布小提琴图')
axes[1].set_xlabel('')
axes[1].set_ylabel('评分')

fig.suptitle('女性主义与非女性主义作品评分分布对比', fontsize=16)
plt.tight_layout()
plt.savefig('分布对比图.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 7. 图4：均值条形图 + 统计检验 ==========
summary_stats = compare_df.groupby('类型')['评分'].agg(['mean', 'std', 'count']).reset_index()
summary_stats['sem'] = summary_stats['std'] / np.sqrt(summary_stats['count'])

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(summary_stats['类型'], summary_stats['mean'], yerr=summary_stats['sem'],
              capsize=10, color=['#FFB6C1', '#87CEEB'], edgecolor='black', alpha=0.8)
ax.set_ylabel('平均评分')
ax.set_title('平均评分对比（含标准误）')

for bar, mean in zip(bars, summary_stats['mean']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')

fem_scores = compare_df[compare_df['类型'] == '女性主义']['评分'].dropna()
nonfem_scores = compare_df[compare_df['类型'] == '非女性主义']['评分'].dropna()
if len(fem_scores) > 1 and len(nonfem_scores) > 1:
    t_stat, p_value = stats.ttest_ind(fem_scores, nonfem_scores, equal_var=False)
    ax.text(0.5, 0.9, f'独立样本t检验 p = {p_value:.4f}', transform=ax.transAxes,
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
elif len(nonfem_scores) == 0:
    ax.text(0.5, 0.9, '无非女性主义评分数据，无法进行t检验', transform=ax.transAxes,
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('均值对比图.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 8. 图5：年度面积图 ==========
fig, ax = plt.subplots(figsize=(12, 6))
years_sorted = yearly_summary['上映年份'].astype(str)
ax.plot(years_sorted, yearly_summary['女性主义平均评分'], color='#DC143C', marker='o', linewidth=2, label='女性主义平均评分')
if not yearly_nonfem.empty:
    ax.plot(years_sorted, yearly_summary['非女性主义平均评分'], color='#000080', marker='s', linestyle='--', linewidth=2, label='非女性主义平均评分')
    ax.fill_between(years_sorted, yearly_summary['女性主义平均评分'], yearly_summary['非女性主义平均评分'],
                    alpha=0.4, color='#4169E1', label='差值区域')
ax.set_xlabel('上映年份')
ax.set_ylabel('平均评分')
ax.set_title('两类作品平均评分年度变化与差异')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('评分年度面积图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n所有图表已生成并保存。")