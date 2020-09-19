import multiprocessing
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from crawl import RAW_DATA_DIR
from utils import tcnv3_to_unicode

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CPU_COUNT = multiprocessing.cpu_count()


def pdf_to_text(pdf_file: str):
    pdf_file_name = pdf_file.split("/")[-1]
    print(f"[PDF] Converting {pdf_file_name}")

    txt_file = pdf_file.split("/")[-1].replace(".pdf", ".txt")
    txt_file = os.path.join(PROCESSED_DATA_DIR, txt_file)
    subprocess.run(f"pdftotext \"{pdf_file}\" \"{txt_file}\"", shell=True, check=True)
    return txt_file


def convert_encoding(txt_file):
    txt_file_name = txt_file.split("/")[-1]
    print(f"[ENCODE] Converting the encoding of {txt_file_name}")
    with open(txt_file) as f:
        tcvn3str = f.read()
    unicode_str = tcnv3_to_unicode(tcvn3str)
    with open(txt_file, "w") as f:
        f.write(unicode_str)


def is_valid_document(pdf_file: str):
    if not pdf_file.endswith(".pdf"):
        return False
    if os.path.getsize(pdf_file) > 15 * 1024 * 1024:
        # too large pdf file -> only images
        return False
    return True


def preprocess_process_file(pdf_file: str):
    if is_valid_document(pdf_file):
        txt_file = pdf_to_text(pdf_file)
        convert_encoding(txt_file)
    else:
        print(f"[SKIP] {pdf_file.split('/')[-1]}")


def preprocess():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        pdf_files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR)]
        futures = [executor.submit(preprocess_process_file, pdf_file) for pdf_file in pdf_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(e)


if __name__ == '__main__':
    preprocess()
