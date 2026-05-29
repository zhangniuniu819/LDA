import pandas as pd
import jieba
import re
import os


# ==================== 1. 加载情感词典 ====================
def load_sentiment_dict(file_path):
    """
    读取 BosonNLP 情感词典，返回 {词: 分数} 的字典。
    词典格式：词\t分数
    """
    sentiment_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                word, score = parts[0], float(parts[1])
                sentiment_dict[word] = score
    return sentiment_dict


# ==================== 2. 设置否定词和程度副词 ====================
# 常用否定词
negation_words = set(['不', '没', '无', '非', '莫', '勿', '未', '否', '别', '休', '甭', '没有'])
# 常用程度副词（可自定义权重）
degree_words = {
    '很': 1.5, '非常': 1.8, '极': 2.0, '极其': 2.0, '格外': 1.6,
    '太': 1.8, '更': 1.5, '更加': 1.5, '最': 2.0, '最为': 2.0,
    '略': 0.5, '稍微': 0.5, '些许': 0.5, '有些': 0.5, '一点': 0.5,
    '挺': 1.3, '相当': 1.5, '蛮': 1.3
}


# ==================== 3. 情感计算函数 ====================
def calculate_sentiment(text, sentiment_dict):
    """
    输入一条评论文本，返回情感得分和标签。
    使用滑动窗口处理否定词（前面2个词内的否定词翻转当前词情感），
    程度副词乘以权重。
    """
    words = list(jieba.cut(text))
    score = 0.0
    i = 0
    n = len(words)

    while i < n:
        word = words[i]
        if word in sentiment_dict:
            word_score = sentiment_dict[word]
            # 向前看至多2个词，检查是否有否定词
            negate = False
            degree = 1.0
            # 检查前面1-2个位置（在边界内）
            for j in range(max(0, i - 2), i):
                if words[j] in negation_words:
                    negate = True
                if words[j] in degree_words:
                    degree *= degree_words[words[j]]  # 可能叠加程度副词
            # 应用否定和程度
            if negate:
                word_score = -word_score
            word_score = word_score * degree
            score += word_score
        i += 1

    # 归一化？不强制，但可以按文本长度做一点调整，避免长文本虚高
    # 这里简单除以有效情感词个数（避免太长文本导致分数过高，可选）
    # 我们直接使用原始总分
    return score


def sentiment_label(score, threshold=0.0):
    if score > threshold:
        return 'positive'
    elif score < -threshold:
        return 'negative'
    else:
        return 'neutral'


# ==================== 4. 主流程 ====================
def main():
    # 路径设置（请根据实际情况修改）
    books_csv = 'doulist_film_comments.csv'  # 书评文件
    dict_file = 'BosonNLP_sentiment_score.txt'  # 情感词典

    # 读取书评数据
    df = pd.read_csv(books_csv, encoding='utf-8')
    print(f"读取到 {len(df)} 条影评")

    # 确认评论文本所在的列名（请根据你的CSV列名修改，常见如 'comment', 'content', 'text'）
    # 这里假设列名为 'comment'，如果不是请替换
    text_column = '评论内容'
    if text_column not in df.columns:
        # 尝试自动猜测
        for col in df.columns:
            if '评论' in col or '内容' in col or 'text' in col or 'comment' in col:
                text_column = col
                break
        else:
            print("无法找到评论文本列，请手动指定 text_column 变量")
            print("当前列名：", df.columns.tolist())
            return

    print(f"使用文本列: {text_column}")

    # 加载情感词典
    sentiment_dict = load_sentiment_dict(dict_file)
    print(f"情感词典加载完成，词条数：{len(sentiment_dict)}")

    # 计算每条评论的情感得分
    print("正在计算情感得分...")
    df['sentiment_score'] = df[text_column].astype(str).apply(
        lambda x: calculate_sentiment(x, sentiment_dict)
    )
    # 生成情感标签
    df['sentiment_label'] = df['sentiment_score'].apply(sentiment_label)

    # 统计概览
    print("\n情感分布：")
    print(df['sentiment_label'].value_counts())
    print(f"平均情感得分: {df['sentiment_score'].mean():.4f}")
    print(f"最高得分: {df['sentiment_score'].max():.4f}")
    print(f"最低得分: {df['sentiment_score'].min():.4f}")

    # 保存结果
    output_file = 'douban_film_comments_with_sentiment.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n情感分析结果已保存至: {output_file}")


if __name__ == '__main__':
    main()