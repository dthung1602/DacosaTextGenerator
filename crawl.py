import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://tulieuvankien.dangcongsan.vn"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CONCURRENT_REQUESTS = 8
ROOT_URLS = [
    {
        "url": "http://tulieuvankien.dangcongsan.vn/c-mac-angghen-lenin-ho-chi-minh/v-i-lenin/tac-pham",
        "max_page": 3
    },
    {
        "url": "http://tulieuvankien.dangcongsan.vn/c-mac-angghen-lenin-ho-chi-minh/ho-chi-minh/tac-pham",
        "max_page": 2
    },
    {
        "url": "http://tulieuvankien.dangcongsan.vn/van-kien-tu-lieu-ve-dang/van-kien-dang-toan-tap",
        "max_page": 4
    }
]


def generate_urls():
    for root_url in ROOT_URLS:
        for page in range(root_url['max_page']):
            yield root_url['url'] + f'?page={page}'


def get_book_detail_urls(page_url):
    response = requests.get(page_url)
    if response.status_code != 200:
        raise Exception(f"Received non-200 response on URL: {page_url}")
    bs = BeautifulSoup(response.text)
    return [BASE_URL + a['href'] for a in bs.select(".booklist .avatar")]


def save_pdf(page_url):
    response = requests.get(page_url)
    if response.status_code != 200:
        raise Exception(f"Received non-200 response on URL: {page_url}")

    bs = BeautifulSoup(response.text)
    pdf_url = BASE_URL + bs.select(".btn-download a")[0]['href']

    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Received non-200 response on URL: {pdf_url}")

    filename = pdf_url.split('/')[-1]
    print(f"Saving to file {filename}")
    with open(os.path.join(RAW_DATA_DIR, filename), 'wb') as f:
        f.write(response.content)


def crawl():
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # crawl book detail pages from book listing pages
        urls = list(generate_urls())
        print(f"----> Start downloading from {len(urls)} pages")
        future_to_url = {executor.submit(get_book_detail_urls, url): url for url in urls}
        book_detail_urls = []

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                book_detail_urls += future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))

        print(f"----> Start downloading {len(book_detail_urls)} books")
        futures = [executor.submit(save_pdf, url) for url in book_detail_urls]

        if not os.path.exists(RAW_DATA_DIR):
            os.makedirs(RAW_DATA_DIR)

        wait(futures)


if __name__ == '__main__':
    crawl()
