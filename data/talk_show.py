import os
import re
import pandas as pd

def process_talk_txt(filepath):
    # 从文件名提取演员名，例如 "小鹿.txt" -> "小鹿"
    basename = os.path.splitext(os.path.basename(filepath))[0]
    actor = basename.strip()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配模式：演员名+数字+冒号，例如 "小鹿1："，"小鹿2："
    # 注意演员名可能和文件名稍有不同，这里用文件名作为前缀更可靠
    pattern = rf'({re.escape(actor)}\d+)\s*[：:]'
    splits = re.split(pattern, content)

    # splits 将是 [开头空白, "小鹿1", "正文1", "小鹿2", "正文2", ...]
    # 我们需要成对提取
    data = []
    for i in range(1, len(splits) - 1, 2):
        label = splits[i]          # e.g., "小鹿1"
        text = splits[i+1].strip() # 该篇内容
        if text:
            data.append({'Actor': actor, 'Label': label, 'Text': text})

    return data

if __name__ == '__main__':
    all_rows = []
    for filename in os.listdir('..'):
        if filename.endswith('.txt'):
            full_path = os.path.join('..', filename)
            print(f'处理文件：{filename}')
            rows = process_talk_txt(full_path)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('talk_show_corpus.csv', index=False, encoding='utf-8-sig')
    print(f'\n全部完成！共提取 {len(df)} 篇稿子，保存至 talk_show_corpus.csv')