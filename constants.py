import os
from enum import Enum

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

CHAR_VOCABULARY = " \n!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~" \
              "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộ" \
              "ỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–"
CHAR_VOCAB_SIZE = len(CHAR_VOCABULARY)
CHAR_VOCAB_MAPPING = {c: i for i, c in enumerate(CHAR_VOCABULARY)}

NEPTUNE_API_TOKEN = os.environ['NEPTUNE_API_TOKEN']
NEPTUNE_PROJECT_NAME = os.environ['NEPTUNE_PROJECT_NAME']
SEED = 20200921

class BookSet(Enum):
    HCM = 'HO CHI MINH'
    LENIN = 'LENIN'
    VK_DANG = 'VK DANG'


class LanguageModelLevel(Enum):
    CHAR_LEVEL = "char"
    WORD_LEVEL = "word"  # not available yet
