import re


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split('\n')


def dbc2sbc(s):
    rs = ''
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs
