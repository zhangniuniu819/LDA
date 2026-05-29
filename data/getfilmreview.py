import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os

# ==================== 你的豆瓣 Cookie，请勿外传 ====================
COOKIE = 'bid=_jEyZFjScvI; dbcl2="222127609:VC5FfDdzVYI"'
# ================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Cookie': COOKIE,
    'Referer': 'https://movie.douban.com',
}

def fetch_comments(movie_id, movie_title, max_pages=3):
    """爬取单部电影的短评，每页20条，max_pages为最大页数"""
    comments = []
    for page in range(max_pages):
        start = page * 20
        url = f'https://movie.douban.com/subject/{movie_id}/comments?start={start}&limit=20&status=P'
        print(f'  正在抓取《{movie_title}》第{page+1}页...', end=' ')
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 403:
                print('403 被限制，请停止或更新Cookie')
                return comments  # 返回已抓到的
            if resp.status_code != 200:
                print(f'状态码 {resp.status_code}')
                continue

            soup = BeautifulSoup(resp.text, 'lxml')
            items = soup.find_all('div', class_='comment-item')
            if not items:
                print('无评论')
                break

            for item in items:
                # 评论文本
                text_tag = item.find('span', class_='short')
                if not text_tag:
                    continue
                text = text_tag.text.strip()
                # 评分（星级在class里）
                rating_tag = item.find('span', class_=re.compile('allstar'))
                rating = rating_tag['class'][0] if rating_tag else ''
                # 时间
                time_tag = item.find('span', class_='comment-time')
                comment_time = time_tag.get('title', '') if time_tag else ''
                # 有用数
                votes_tag = item.find('span', class_='votes')
                votes = votes_tag.text.strip() if votes_tag else '0'

                comments.append({
                    '电影名': movie_title,
                    '电影豆瓣ID': movie_id,
                    '评论内容': text,
                    '评论评分': rating,
                    '评论时间': comment_time,
                    '有用数': votes
                })
            print(f'获取 {len(items)} 条')
            time.sleep(random.uniform(5, 8))  # 必须慢

        except Exception as e:
            print(f'出错：{e}')
            time.sleep(15)
            continue
    return comments

if __name__ == '__main__':
    # 1. 读取电影列表
    df_movies = pd.read_csv('doulist_feminism_movies2 .csv')
    print(f"共读取 {len(df_movies)} 部电影")

    # 2. 筛选核心电影（可选，强烈建议！）
    # 只抓评分>8.5 且 评价人数>10000的，避免反爬
    # 如果你的CSV里没有“评价人数”列，可以先注释掉这两行
    # df_movies = df_movies[(df_movies['评分'] > 8.5) & (df_movies['评价人数'] > 10000)]
    # print(f"筛选后剩余 {len(df_movies)} 部核心电影")

    all_comments = []
    for idx, row in df_movies.iterrows():
        title = row['电影名']
        movie_url = row['豆瓣链接']
        # 从豆瓣链接中提取电影ID
        match = re.search(r'subject/(\d+)/', movie_url)
        if not match:
            print(f'跳过《{title}》，无法提取ID')
            continue
        movie_id = match.group(1)
        print(f'\n[{idx+1}/{len(df_movies)}] 开始爬取《{title}》')
        comments = fetch_comments(movie_id, title, max_pages=3)  # 每部最多60条短评
        all_comments.extend(comments)
        # 每部电影之间休息10-15秒
        time.sleep(random.uniform(10, 15))

    # 3. 保存
    if all_comments:
        df_comments = pd.DataFrame(all_comments)
        df_comments.to_csv('doulist_film_comments.csv', index=False, encoding='utf-8-sig')
        print(f'\n全部完成！共保存 {len(df_comments)} 条评论')
    else:
        print('未获取到任何评论')