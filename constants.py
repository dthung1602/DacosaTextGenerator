import os
from enum import Enum

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
VALID_CHARS = " \n!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~" \
              "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộ" \
              "ỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–"
CHAR_MAPPING = {c: i for i, c in enumerate(VALID_CHARS)}


class BookSet(Enum):
    HCM = 'HO CHI MINH'
    LENIN = 'LENIN'
    VK_DANG = 'VK DANG'


class SampleStrategy(Enum):
    BY_LINE = 'BY_LINE'
    BY_LENGTH = 'BY_LENGTH'


class LanguageModelLevel(Enum):
    CHAR_LEVEL = "CHAR_LEVEL"
    WORD_LEVEL = "WORD_LEVEL"  # not available yet
