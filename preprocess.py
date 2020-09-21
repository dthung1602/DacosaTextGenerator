import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from constants import PROCESSED_DATA_DIR, RAW_DATA_DIR, CHAR_VOCAB_MAPPING, BookSet
from utils import tcnv3_to_unicode

CPU_COUNT = multiprocessing.cpu_count()

BOOKSET_TO_REMOVE_PART = {
    BookSet.HCM: ["mục lục"],
    BookSet.LENIN: ["mà v. i. lê-nin đã trích dẫn", "mục lục"],
    BookSet.VK_DANG: ["mục lục"]
}

PAGE_HEADERS = [
    "V. I. L ê - n i n".lower(),
    "Hồ CHí MINH TOàN TậP".lower(),
    "Văn kiện đảng toàn tập".lower(),
]


def pdf_to_text(pdf_file: str):
    pdf_file_name = pdf_file.split("/")[-1]
    print(f"[PDF] Converting {pdf_file_name}")

    txt_file = pdf_file.split("/")[-1].replace(".pdf", ".txt")
    txt_file = os.path.join(PROCESSED_DATA_DIR, txt_file)
    subprocess.run(f'pdftotext -f 5 "{pdf_file}" "{txt_file}"', shell=True, check=True)
    return txt_file


def remove_last_pages(file_content: str, file_name: str) -> str:
    bookset = None
    for bs in [BookSet.LENIN, BookSet.VK_DANG, BookSet.HCM]:
        if file_name.upper().startswith(bs.value):
            bookset = bs
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
    return " ".join(new_lines)


def convert_encoding(txt_file):
    txt_file_name = txt_file.split("/")[-1]
    print(f"[ENCODE] Converting the encoding of {txt_file_name}")
    with open(txt_file) as f:
        tcvn3str = f.read()
    unicode_str = tcnv3_to_unicode(tcvn3str)
    unicode_str = remove_last_pages(unicode_str, txt_file_name)
    unicode_str = remove_too_short_lines(unicode_str)
    with open(txt_file, "w") as f:
        f.write(unicode_str)


def is_valid_document(pdf_file: str):
    if not pdf_file.endswith(".pdf"):
        return False
    if pdf_file.split("/")[-1][0].islower():
        return False
    if os.path.getsize(pdf_file) > 15 * 1024 * 1024:
        # too large pdf file -> only images
        return False
    return True


def preprocess_process_file(pdf_file: str):
    if is_valid_document(pdf_file):
        txt_file = pdf_to_text(pdf_file)
        convert_encoding(txt_file)
        character_encoding(txt_file)
        word_encoding(txt_file)
    else:
        print(f"[SKIP] {pdf_file.split('/')[-1]}")


def word_encoding(txt_file: str):
    # use .word.npy
    pass


def character_encoding(txt_file: str):
    txt_file_name = txt_file.split("/")[-1]
    print(f"[CHAR] Character-based encoding {txt_file_name}")
    with open(txt_file) as f:
        txt = f.read()
    arr = np.array([CHAR_VOCAB_MAPPING[c] for c in txt])
    numpy_file = txt_file.replace(".txt", ".char.npy")
    np.save(numpy_file, arr)


def preprocess():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    pdf_files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR)]
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        future_to_pdffile = {executor.submit(preprocess_process_file, pdf_file): pdf_file for pdf_file in pdf_files}
        for future in as_completed(future_to_pdffile):
            try:
                future.result()
            except Exception as e:
                pdf = future_to_pdffile[future].split("/")[-1]
                print(f"ERROR processing file {pdf}", file=sys.stderr)
                print(e, file=sys.stderr)


if __name__ == '__main__':
    preprocess()
