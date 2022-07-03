def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def read_byte_len(char):
    if len(char.encode()) > 1:
        return 2
    else:
        return 1

def convert_segment_to_token_box(text, bbox):
    byte_len = sum([read_byte_len(c) for c in text])
    width_per_byte = (bbox[2] - bbox[0]) / byte_len
    words = []
    accu_byte_len = 0
    for t in text:
        x0 = bbox[0] + accu_byte_len * width_per_byte
        x1 = bbox[0] + (accu_byte_len + read_byte_len(t)) * width_per_byte
        accu_byte_len = accu_byte_len + read_byte_len(t)
        words.append({'text': t, 'box': [x0, bbox[1], x1, bbox[3]]})
    return words


if __name__ == '__main__':
    text = 'aa长盛缘超市双台子店'
    bbox = [265, 182, 597, 231]
    convert_segment_to_token_box(text, bbox)
