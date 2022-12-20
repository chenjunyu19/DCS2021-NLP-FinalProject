import json
import os

import torch
from torch.utils.data import DataLoader

import model as M
import utils

CONFIG = utils.read_config()

# 读取 词语-id 映射
with open(os.path.join(CONFIG['dataDir'], 'word2id.json'), 'r') as f:
    word2id = json.load(f)
id2word = {}
with open(os.path.join(CONFIG['dataDir'], 'id2word.json'), 'r') as f:
    # JSON 的键是字符串，需要转换回整数
    for k, v in json.load(f).items():
        id2word[int(k)] = v

# 读取测试集
lines, words = utils.read_data('test_en.txt', CONFIG['tokenizer'])
ds_test = M.Dataset(word2id, id2word, lines)
dl_test = DataLoader(ds_test)

# 创建 RNN 模型
VOCAB_SIZE = len(word2id)
model = M.RNNModel('RNN_TANH', VOCAB_SIZE, CONFIG['embeddingSize'],
                   nhid=512, dropout=0.5)
if CONFIG['useCUDA']:
    model = model.cuda()

# 载入训练结果
print('Which state dict do you want to use?')
fname = input('Type `best`, `last` or epoch number: ')
model.load_state_dict(torch.load(os.path.join(
    CONFIG['dataDir'], f'state_dict_{fname}.th')))

# 进行测试
model.eval()
with torch.no_grad():
    hidden = model.init_hidden(requires_grad=False)
    # 将数据按batch输入
    for i, batch in enumerate(dl_test):
        data, target = batch
        if CONFIG['useCUDA']:
            data, target = data.cuda(), target.cuda()

        hidden = M.repackage_hidden(hidden)

        with torch.no_grad():
            output, hidden = model(data, hidden)
        if model.rnn_type == "LSTM":
            # LSTM 输出 output, (h_n, c_n)
            # h_n 保存着 RNN 最后一个时间步的隐状态。
            # c_n 保存着 RNN 最后一个时间步的细胞状态。
            hidden = hidden[0]
        decoded = model.decoder(hidden.view(
            hidden.size(0) * hidden.size(1), hidden.size(2)))
        result = {}
        for i, score in enumerate(decoded[0]):
            result[id2word[i]] = float(score)
        print('=' * 16)
        print(ds_test.get_words_by_ids([int(i)
              for i in list(batch[0][0])]) + ':')
        for k, v in sorted(result.items(), key=lambda s: -s[1])[:10]:
            print(k, v)
