import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# ==================== 你的豆瓣 Cookie，请勿外传 ====================
COOKIE = 'bid=_jEyZFjScvI; dbcl2="222127609:VC5FfDdzVYI"'
# ================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Cookie': COOKIE,
    'Referer': 'https://www.douban.com',
}

def fetch_url(url, max_retries=3):
    """带重试的请求，超时时间为30秒"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)  # 超时改为30秒
            return resp
        except requests.exceptions.ReadTimeout:
            print(f'  请求超时，第{attempt+1}次重试...')
            time.sleep(5 * (attempt + 1))  # 阶梯等待
        except Exception as e:
            print(f'  请求出错：{e}，第{attempt+1}次重试...')
            time.sleep(5 * (attempt + 1))
    return None

def fetch_doulist_movies(doulist_id, max_pages=15):
    movies = []
    for page in range(max_pages):
        start = page * 25
        url = f'https://www.douban.com/doulist/{doulist_id}/?start={start}&sort=seq&playable=0&sub_type='
        print(f'  正在抓取第 {page+1} 页...', end=' ')

        resp = fetch_url(url)
        if resp is None:
            print('多次重试后仍失败，跳过本页')
            continue
        if resp.status_code == 404:
            print('404 页面不存在')
            break
        elif resp.status_code == 403:
            print('403 被限制，请更新 Cookie 或停止一段时间')
            break
        elif resp.status_code != 200:
            print(f'状态码 {resp.status_code}')
            continue

        soup = BeautifulSoup(resp.text, 'lxml')
        items = soup.select('div.article div.doulist-item')
        if not items:
            print('没有更多条目了，可能已抓完')
            break

        for item in items:
            # --- 电影名和链接 ---
            title_div = item.select_one('div.title a')
            if not title_div:
                continue
            title = title_div.text.strip()
            movie_url = title_div.get('href', '')

            # --- 基本信息 ---
            abstract_div = item.select_one('div.abstract')
            abstract_text = abstract_div.text.strip() if abstract_div else ''

            year = ''
            years = re.findall(r'(\d{4})', abstract_text)
            if years:
                year = years[0]

            director = ''
            dir_match = re.search(r'导演:(.*?)(?:主演|类型|制片|年份|\d{4}|$)', abstract_text)
            if dir_match:
                director = dir_match.group(1).strip()

            genres = ''
            genre_match = re.search(r'类型:(.*?)(?:制片|年份|\d{4}|$)', abstract_text)
            if genre_match:
                genres = genre_match.group(1).strip()

            # --- 评分 ---
            rating_div = item.select_one('span.rating_nums')
            if not rating_div:
                rating_match = re.search(r'(\d\.\d)', str(item))
                rating = rating_match.group(1) if rating_match else '0.0'
            else:
                rating = rating_div.text.strip()

            # --- 评语 ---
            comment_block = item.select_one('blockquote.modify-time')
            comment = comment_block.text.strip() if comment_block else ''

            movies.append({
                '电影名': title,
                '导演': director,
                '上映年份': year,
                '类型': genres,
                '评分': rating,
                '评语': comment,
                '豆瓣链接': movie_url,
                '来源豆列': f'女性电影TOP300 (ID: {doulist_id})'
            })

        print(f'√ 获取 {len(items)} 条')
        time.sleep(random.uniform(5, 8))  # 延时也稍微加长

    return movies

if __name__ == '__main__':
    print('开始爬取豆瓣豆列：女性电影TOP300')
    doulist_id = '161498552'
    all_movies = fetch_doulist_movies(doulist_id, max_pages=15)

    df = pd.DataFrame(all_movies)
    if not df.empty:
        df = df[df['上映年份'].astype(str).str.isdigit()]
        df = df[(df['上映年份'].astype(int) >= 2005) & (df['上映年份'].astype(int) <= 2025)]
        output_filename = 'doulist_feminism_movies.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f'\n完成！共保存 {len(df)} 部电影元数据到 {output_filename}')
    else:
        print('未获取到任何数据，请检查 Cookie 或网络')