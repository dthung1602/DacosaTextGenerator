from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

from constants import *

CONCURRENT_REQUESTS = 8


class AbstractDataSource(ABC):
    @abstractmethod
    def download(self, url):
        pass

    @abstractmethod
    def generate_urls(self):
        pass


class DCSSource(AbstractDataSource):
    BASE_URL = "http://tulieuvankien.dangcongsan.vn"
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

    def generate_urls(self):
        book_listing_urls = self.__generate_book_listing_urls()
        book_detail_urls = []
        for blu in book_listing_urls:
            book_detail_urls += self.__generate_book_detail_urls(blu)
        return book_detail_urls

    def __generate_book_listing_urls(self):
        for root_url in self.ROOT_URLS:
            for page in range(root_url['max_page']):
                yield root_url['url'] + f'?page={page}'

    def __generate_book_detail_urls(self, book_listing_url):
        response = requests.get(book_listing_url)
        if response.status_code != 200:
            raise Exception(f"Received non-200 response on URL: {book_listing_url}")
        bs = BeautifulSoup(response.text, features="lxml")
        return [self.BASE_URL + a['href'] for a in bs.select(".booklist .avatar")]

    def download(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Received non-200 response on URL: {url}")

        bs = BeautifulSoup(response.text)
        pdf_url = self.BASE_URL + bs.select(".btn-download a")[0]['href']

        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Received non-200 response on URL: {pdf_url}")

        filename = pdf_url.split('/')[-1].replace("-", " - ").replace("  ", " ")
        print(f"Saving to file {filename}")
        with open(os.path.join(RAW_DATA_DIR, filename), 'wb') as f:
            f.write(response.content)


class TriThucLuanSource(AbstractDataSource):
    URL_TEMPLATE = "https://trithuclyluan.com/wp-content/uploads/2020/04/mactap-{}.pdf"

    def generate_urls(self):
        return [self.URL_TEMPLATE.format(volume_no) for volume_no in range(1, 51)]

    def download(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Received non-200 response on URL: {url}")

        volume_no = url[60:-4]
        filename = f"CAC MAC TOAN TAP - TAP {volume_no}.pdf"
        pdf_file = os.path.join(RAW_DATA_DIR, filename)

        print(f"Saving to file {filename}")
        with open(pdf_file, "wb") as f:
            f.write(response.content)


def main():
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # data is also download manually from https://marxists.architexturez.net/vietnamese/cac-tac-gia-khac.htm
    dcs_source = DCSSource()
    ttl_source = TriThucLuanSource()

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # crawl book detail pages from book listing pages
        dcs_urls = list(dcs_source.generate_urls())
        ttl_urls = list(ttl_source.generate_urls())

        print(f"----> Start downloading from {len(dcs_urls) + len(ttl_urls)} pages <----")
        future_to_url = {
            **{executor.submit(dcs_source.download, url): url for url in dcs_urls},
            **{executor.submit(ttl_source.download, url): url for url in ttl_urls}
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))

    print("----> Done! <----")


if __name__ == '__main__':
    main()
