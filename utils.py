from typing import Sized

from constants import VALID_CHARS

TCVN3 = ["Aµ", "A¸", "¢", "A·", "EÌ", "EÐ", "£", "I×", "IÝ", "Oß",
         "Oã", "¤", "Oâ", "Uï", "Uó", "Yý", "µ", "¸", "©", "·",
         "Ì", "Ð", "ª", "×", "Ý", "ß", "ã", "«", "â", "ï",
         "ó", "ý", "¡", "¨", "§", "®", "IÜ", "Ü", "Uò", "ò",
         "¥", "¬", "¦", "−", "A¹", "¹", "A¶", "¶", "¢Ê", "Ê",
         "¢Ç", "Ç", "¢È", "È", "¢É", "É", "¢Ë", "Ë", "¡¾", "¾",
         "¡»", "»", "¡¼", "¼", "¡½", "½", "¡Æ", "Æ", "EÑ", "Ñ",
         "EÎ", "Î", "EÏ", "Ï", "£Õ", "Õ", "£Ò", "Ò", "£Ó", "Ó",
         "£Ô", "Ô", "£Ö", "Ö", "IØ", "Ø", "IÞ", "Þ", "Oä", "ä",
         "Oá", "á", "¤è", "è", "¤å", "å", "¤æ", "æ", "¤ç", "ç",
         "¤é", "é", "¥í", "í", "¥ê", "ê", "¥ë", "ë", "¥ì", "ì",
         "¥î", "î", "Uô", "ô", "Uñ", "ñ", "¦ø", "ø", "¦õ", "õ",
         "¦ö", "ö", "¦÷", "÷", "¦ù", "ù", "Yú", "ú", "Yþ", "þ",
         "Yû", "û", "Yü", "ü", ".", "­"]

UNICODE = ["À", "Á", "Â", "Ã", "È", "É", "Ê", "Ì", "Í", "Ò",
           "Ó", "Ô", "Õ", "Ù", "Ú", "Ý", "à", "á", "â", "ã",
           "è", "é", "ê", "ì", "í", "ò", "ó", "ô", "õ", "ù",
           "ú", "ý", "Ă", "ă", "Đ", "đ", "Ĩ", "ĩ", "Ũ", "ũ",
           "Ơ", "ơ", "Ư", "ư", "Ạ", "ạ", "Ả", "ả", "Ấ", "ấ",
           "Ầ", "ầ", "Ẩ", "ẩ", "Ẫ", "ẫ", "Ậ", "ậ", "Ắ", "ắ",
           "Ằ", "ằ", "Ẳ", "ẳ", "Ẵ", "ẵ", "Ặ", "ặ", "Ẹ", "ẹ",
           "Ẻ", "ẻ", "Ẽ", "ẽ", "Ế", "ế", "Ề", "ề", "Ể", "ể",
           "Ễ", "ễ", "Ệ", "ệ", "Ỉ", "ỉ", "Ị", "ị", "Ọ", "ọ",
           "Ỏ", "ỏ", "Ố", "ố", "Ồ", "ồ", "Ổ", "ổ", "Ỗ", "ỗ",
           "Ộ", "ộ", "Ớ", "ớ", "Ờ", "ờ", "Ở", "ở", "Ỡ", "ỡ",
           "Ợ", "ợ", "Ụ", "ụ", "Ủ", "ủ", "Ứ", "ứ", "Ừ", "ừ",
           "Ử", "ử", "Ữ", "ữ", "Ự", "ự", "Ỳ", "ỳ", "Ỵ", "ỵ",
           "Ỷ", "ỷ", "Ỹ", "ỹ", ".", "ư"]

CHARS_TO_REPLACE = {
    "": "",
    " ư ": " - ",
    " ư\n": " -\n",
    "\nư ": "\n- ",
    "”": "\"",
    "“": "\"",
    "_______________": "",
    "―": "-",
    "⎯": "-",
    "…": "...",
    "`": "",
    "\t": " ",
}


def tcnv3_to_unicode(str_original: str) -> str:
    source_charset = TCVN3
    target_charset = UNICODE

    map_length = len(source_charset)
    for number in range(map_length):
        str_original = str_original.replace(source_charset[number], f"[::{number}::]")

    for number in range(map_length):
        str_original = str_original.replace(f"[::{number}::]", target_charset[number])

    for char, replacement in CHARS_TO_REPLACE.items():
        str_original = str_original.replace(char, replacement)

    new_str = ""
    for char in str_original:
        if char in VALID_CHARS:
            new_str += char

    return new_str


def chunk(array: Sized, chunk_size: int) -> list:
    chunk_count = int(len(array) / chunk_size) + 1
    return [
        array[i * chunk_size: i + chunk_size] for i in range(chunk_count)
    ]
