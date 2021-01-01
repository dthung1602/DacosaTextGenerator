import multiprocessing
import subprocess

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer

from constants import *

CPU_COUNT = multiprocessing.cpu_count()


def get_plain_file_name(file_name: str) -> str:
    return file_name.split("/")[-1].split(".")[0]


def create_dataset(txt_file: str):
    TokenDataset(
        txt_file,
        merges_file=AI_TEXT_GEN_MERGES_FILE,
        vocab_file=AI_TEXT_GEN_VOCAB_FILE,
        save_cache=True,
        cache_destination=os.path.join(AI_TEXT_GEN_TOKENIZED_DIR, get_plain_file_name(txt_file) + ".tar.gz"),
        block_size=AI_TEXT_GEN_BLOCK_SIZE
    )


def combine_txt_files():
    processes = []
    for bookset in BookSet:
        combined_file = f"_{bookset.value.upper().replace(' ', '')}.TXT"
        pattern = bookset.value.replace(' ', '\\ ') + '*'
        proc = subprocess.Popen(f"cat {pattern} > {combined_file}", shell=True, cwd=PROCESSED_DATA_DIR)
        processes.append(proc)

    [proc.wait() for proc in processes]

    subprocess.Popen(f"cat _* > _ALL.TXT", shell=True, cwd=PROCESSED_DATA_DIR).wait()


def main():
    combine_txt_files()

    train_tokenizer(
        os.path.join(PROCESSED_DATA_DIR, "_ALL.TXT"),
        vocab_size=AI_TEXT_GEN_VOCAB_SIZE,
        save_path=AI_TEXT_GEN_TOKENIZED_DIR
    )

    txt_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".TXT")]
    for f in txt_files:
        create_dataset(f)


if __name__ == '__main__':
    main()
