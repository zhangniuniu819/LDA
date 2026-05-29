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
    'Referer': 'https://book.douban.com',
}

def fetch_book_comments(book_id, book_title, max_pages=1):
    """爬取单本书的短评，默认只爬1页（20条）"""
    comments = []
    for page in range(max_pages):
        start = page * 20
        url = f'https://book.douban.com/subject/{book_id}/comments/?start={start}&limit=20&status=P'
        print(f'  正在抓取《{book_title}》第{page+1}页...', end=' ')
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 403:
                print('403 被限制，请停止或更新Cookie')
                return comments
            if resp.status_code == 404:
                print('404 页面不存在')
                return comments
            if resp.status_code != 200:
                print(f'状态码 {resp.status_code}')
                continue

            soup = BeautifulSoup(resp.text, 'lxml')
            items = soup.find_all('div', class_='comment')
            if not items:
                print('无评论')
                break

            for item in items:
                text_tag = item.find('span', class_='short')
                if not text_tag:
                    continue
                text = text_tag.text.strip()

                rating_tag = item.find('span', class_=re.compile('allstar'))
                rating = rating_tag['class'][0] if rating_tag else ''

                time_tag = item.find('span', class_='comment-time')
                comment_time = time_tag.get('title', '') if time_tag else ''

                votes_tag = item.find('span', class_='votes')
                votes = votes_tag.text.strip() if votes_tag else '0'

                comments.append({
                    '书名': book_title,
                    '书籍豆瓣ID': book_id,
                    '评论内容': text,
                    '评论评分': rating,
                    '评论时间': comment_time,
                    '有用数': votes
                })

            print(f'获取 {len(items)} 条')
            time.sleep(random.uniform(4, 7))

        except Exception as e:
            print(f'出错：{e}')
            time.sleep(15)
            continue
    return comments


if __name__ == '__main__':
    df_books = pd.read_csv('douban_feminism_books_metadata.csv')
    print(f"共读取 {len(df_books)} 本书")

    all_comments = []
    for idx, row in df_books.iterrows():
        title = row['书名']
        link = row['豆瓣链接']
        match = re.search(r'subject/(\d+)', link)
        if not match:
            print(f'跳过《{title}》，无法提取ID')
            continue
        book_id = match.group(1)

        print(f'\n[{idx+1}/{len(df_books)}] 开始爬取《{title}》')
        comments = fetch_book_comments(book_id, title, max_pages=1)  # 每本1页
        all_comments.extend(comments)

        time.sleep(3)   # 每本书之间休息 3 秒

        # 每20本休息5分钟
        if (idx + 1) % 20 == 0:
            print(f'\n===== 已爬取 {idx+1} 本，休息30s =====')
            time.sleep(30)

    if all_comments:
        df_comments = pd.DataFrame(all_comments)
        df_comments.to_csv('douban_books_comments.csv', index=False, encoding='utf-8-sig')
        print(f'\n全部完成！共保存 {len(df_comments)} 条评论到 douban_books_comments.csv')
    else:
        print('未获取到任何评论')