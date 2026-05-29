import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# ==================== 已填入你的Cookie，请勿外传 ====================
COOKIE = 'bid=_jEyZFjScvI; dbcl2="222127609:VC5FfDdzVYI"'
# ====================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Cookie': COOKIE,
    'Referer': 'https://book.douban.com',
}

def fetch_books_by_tag(tag, max_pages=10):
    """爬取指定标签的书籍列表，每页20本，max_pages为最大翻页数"""
    books = []
    for page in range(max_pages):
        start = page * 20
        url = f'https://book.douban.com/tag/{tag}?start={start}&type=T'
        print(f'  正在抓取 {tag} 第 {page+1} 页...', end=' ')
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 404:
                print('404 页面不存在')
                break
            elif resp.status_code == 403:
                print('403 被限制，可能需要更新Cookie或降低频率')
                break
            elif resp.status_code != 200:
                print(f'状态码 {resp.status_code}，重试可能有效')
                continue

            soup = BeautifulSoup(resp.text, 'lxml')
            items = soup.find_all('li', class_='subject-item')
            if not items:
                print('无书籍条目，可能已到最后一页')
                break

            for item in items:
                # --- 书名 ---
                title_tag = item.find('a', title=True)
                if not title_tag:
                    title_tag = item.find('h2').find('a') if item.find('h2') else None
                title = title_tag['title'].strip() if title_tag and title_tag.get('title') else ''
                if not title:
                    continue

                book_url = title_tag['href'] if title_tag else ''

                # --- 信息行 ---
                info_div = item.find('div', class_='info')
                pub_info = ''
                if info_div:
                    pub_div = info_div.find('div', class_='pub')
                    if pub_div:
                        pub_info = pub_div.text.strip()

                # 从pub中提取出版年（取第一个四位数字）
                year = ''
                if pub_info:
                    years = re.findall(r'(\d{4})', pub_info)
                    if years:
                        year = years[0]

                # --- 作者 ---
                author = ''
                if pub_info and '/' in pub_info:
                    parts = pub_info.split('/')
                    if parts:
                        author = parts[0].strip()

                # --- 评分 ---
                rating_tag = item.find('span', class_='rating_nums')
                rating = rating_tag.text.strip() if rating_tag else '0.0'

                # --- 评价人数 ---
                pl_tag = item.find('span', class_='pl')
                people_num = ''
                if pl_tag:
                    nums = re.findall(r'\d+', pl_tag.text)
                    if nums:
                        people_num = nums[0]

                # --- 标签 ---
                tag_list = []
                ft_div = item.find('div', class_='ft')
                if ft_div:
                    ft_tags = ft_div.find_all('a')
                    for a in ft_tags:
                        if a.text.strip():
                            tag_list.append(a.text.strip())
                tags_str = ';'.join(tag_list)

                books.append({
                    '书名': title,
                    '作者': author,
                    '出版年': year,
                    '评分': rating,
                    '评价人数': people_num,
                    '标签': tags_str,
                    '豆瓣链接': book_url,
                    '爬取标签': tag
                })

            print(f'获取 {len(items)} 条')
            time.sleep(random.uniform(3, 6))

        except Exception as e:
            print(f'出错：{e}')
            time.sleep(10)
            continue

    return books

if __name__ == '__main__':
    tag_list = ['女性主义', '性别研究', '女权', '厌女', '性别平等']
    all_books = []

    for tag in tag_list:
        print(f'\n开始爬取标签：{tag}')
        books = fetch_books_by_tag(tag, max_pages=10)
        print(f'{tag} 共获取 {len(books)} 本')
        all_books.extend(books)
        time.sleep(random.uniform(5, 8))

    df = pd.DataFrame(all_books)
    if not df.empty:
        df.drop_duplicates(subset=['书名', '作者'], inplace=True)
        # 只保留2005-2025年出版的书籍
        df = df[df['出版年'].astype(str).str.isdigit()]
        df = df[(df['出版年'].astype(int) >= 2005) & (df['出版年'].astype(int) <= 2025)]

        df.to_csv('douban_feminism_books_metadata.csv', index=False, encoding='utf-8-sig')
        print(f'\n完成！共保存 {len(df)} 条去重书籍元数据到 douban_feminism_books_metadata.csv')
    else:
        print('未获取到任何数据，请检查Cookie或网络')