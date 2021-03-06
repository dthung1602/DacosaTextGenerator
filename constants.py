import os
from enum import Enum

ROOT_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

AI_TEXT_GEN_TOKENIZED_DIR = os.path.join(PROCESSED_DATA_DIR, "tokenized")
AI_TEXT_GEN_VOCAB_FILE = os.path.join(AI_TEXT_GEN_TOKENIZED_DIR, "vocab.json")
AI_TEXT_GEN_MERGES_FILE = os.path.join(AI_TEXT_GEN_TOKENIZED_DIR, "merges.txt")
AI_TEXT_GEN_VOCAB_SIZE = 10000
AI_TEXT_GEN_BLOCK_SIZE = 1024

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = os.environ.get('NEPTUNE_PROJECT_NAME')
SEED = 20200921


class BookSet(Enum):
    HCM = 'HO CHI MINH'
    LENIN = 'LENIN'
    MAC = 'CAC MAC'
    VK_DANG = 'VK Dang'
    OTHER = 'OTHER'
