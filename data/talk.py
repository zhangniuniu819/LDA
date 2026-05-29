import os
import re
import pandas as pd

def process_talk_txt(filepath):
    # 从文件名提取演员名，例如 "杨笠.txt" -> "杨笠"
    basename = os.path.splitext(os.path.basename(filepath))[0]
    actor = basename.strip()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配模式：演员名+数字+冒号，例如 "杨笠1："，"杨笠2："
    pattern = rf'({re.escape(actor)}\d+)\s*[：:]'
    splits = re.split(pattern, content)

    # splits 将是 [开头空白, "杨笠1", "正文1", "杨笠2", "正文2", ...]
    data = []
    for i in range(1, len(splits) - 1, 2):
        label = splits[i]          # e.g., "杨笠1"
        text = splits[i+1].strip() # 该篇内容
        if text:
            data.append({'Actor': actor, 'Label': label, 'Text': text})

    return data

if __name__ == '__main__':
    filename = '杨笠.txt'
    full_path = os.path.join('.', filename)
    print(f'处理文件：{filename}')
    rows = process_talk_txt(full_path)

    df = pd.DataFrame(rows)
    df.to_csv('talk_show_corpus.csv', index=False, encoding='utf-8-sig')
    print(f'\n完成！共提取 {len(df)} 篇稿子，保存至 talk_show_corpus.csv')