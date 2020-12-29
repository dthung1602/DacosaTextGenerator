import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import PROCESSED_DATA_DIR, RAW_DATA_DIR, BookSet
from utils import tcnv3_to_unicode

CPU_COUNT = multiprocessing.cpu_count()

BOOKSET_TO_REMOVE_PART = {
    BookSet.HCM: ["mục lục"],
    BookSet.LENIN: ["mà v. i. lê-nin đã trích dẫn", "mục lục"],
    BookSet.VK_DANG: ["mục lục"],
    BookSet.OTHER: ["Chú thích:"],
    BookSet.MAC: ["CÁC BẢN CHỈ DẪN"],
}

PAGE_HEADERS = [
    "V. I. L ê - n i n".lower(),
    "Hồ CHí MINH TOàN TậP".lower(),
    "Văn kiện đảng toàn tập".lower(),
]


def get_bookset_by_file_name(file_name: str) -> BookSet:
    for bs in [BookSet.LENIN, BookSet.VK_DANG, BookSet.HCM, BookSet.MAC, BookSet.OTHER]:
        if file_name.upper().startswith(bs.value):
            return bs


def pdf_to_text(pdf_file: str):
    pdf_file_name = pdf_file.split("/")[-1]
    print(f"[PDF] Converting {pdf_file_name}")

    txt_file = pdf_file.split("/")[-1].replace(".pdf", ".txt")
    txt_file = os.path.join(PROCESSED_DATA_DIR, txt_file)
    subprocess.run(f'pdftotext "{pdf_file}" "{txt_file}"', shell=True, check=True)
    return txt_file


def remove_last_pages(file_content: str, file_name: str) -> str:
    bookset = get_bookset_by_file_name(file_name)
    remove_parts = BOOKSET_TO_REMOVE_PART[bookset]
    for part in remove_parts:
        try:
            idx = file_content.lower().index(part)
            file_content = file_content[:idx]
        except:
            print(f"> NOT CONTAIN: {file_name} - {part}")
    return file_content


def remove_too_short_lines(file_content: str) -> str:
    lines = file_content.split("\n")
    new_lines = []
    for line in lines:
        if len(line.split(" ")) >= 3 and line.lower() not in PAGE_HEADERS:
            new_lines.append(line)
    return "\n".join(new_lines)


def convert_encoding(txt_file):
    txt_file_name = txt_file.split("/")[-1]
    bookset = get_bookset_by_file_name(txt_file_name)
    print(f"[ENCODE] Converting the encoding of {txt_file_name}")
    with open(txt_file) as f:
        content = f.read()
    if bookset is not BookSet.OTHER:
        content = tcnv3_to_unicode(content)
    content = remove_last_pages(content, txt_file_name)
    content = remove_too_short_lines(content)
    with open(txt_file, "w") as f:
        f.write(content)


def is_valid_document(pdf_file: str):
    if not pdf_file.endswith(".pdf"):
        return False
    if pdf_file.split("/")[-1][0].islower():
        return False
    # if os.path.getsize(pdf_file) > 15 * 1024 * 1024:
    #     # too large pdf file -> only images
    #     return False
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

    pdf_files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR)]
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        future_to_pdffile = {executor.submit(preprocess_process_file, pdf_file): pdf_file for pdf_file in pdf_files}
        # if pdf_file.split("/")[-1].startswith(BookSet.OTHER.value)}
        for future in as_completed(future_to_pdffile):
            try:
                future.result()
            except Exception as e:
                pdf = future_to_pdffile[future].split("/")[-1]
                print(f"ERROR processing file {pdf}", file=sys.stderr)
                print(e, file=sys.stderr)


if __name__ == '__main__':
    preprocess()
