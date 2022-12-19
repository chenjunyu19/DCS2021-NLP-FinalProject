import re

# 用于去除标点的正则
PUNC = r'[.!?/_,$%^*()+"\'+~@#%&]'
SEP = re.compile(PUNC + r'*\s+')
PREFIX = re.compile('^' + PUNC + '+')


def read_data(filename: str):
    "按行读取数据集文件，分词、去除标点后返回 lines, words"

    lines = []
    words = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # 数据集中有些 <br />，替换为空格
            # 先用标点+空格分割，然后去除前缀标点，转为小写
            line = [PREFIX.sub('', word).lower() for word
                    in SEP.split(line.replace('<br />', ' '))
                    if len(PREFIX.sub('', word))]
            lines.append(line)
            # 添加词语到词表
            for word in line:
                words.add(word)
    return lines, words


def make_map(words: set):
    "根据词表，返回双向 词语-id 映射 word2id, id2word"

    word2id = {'<pad>': 0}
    id2word = {0: '<pad>'}

    # 生成 词语-id 映射
    for i, word in enumerate(list(words)):
        word2id[word] = i+1
        id2word[i+1] = word
    
    word2id['<unk>'] = len(word2id)
    id2word[len(id2word)] = '<unk>'

    return word2id, id2word
