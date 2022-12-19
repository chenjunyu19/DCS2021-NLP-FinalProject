import json

import torch
from torch.utils.data import DataLoader

import model as M
import utils

with open('word2id.json', 'r') as f:
    word2id = json.load(f)
id2word = {}
with open('id2word.json', 'r') as f:
    for k, v in json.load(f).items():
        id2word[int(k)] = v

USE_CUDA = torch.cuda.is_available()
VOCAB_SIZE = len(word2id)
BATCH_SIZE = 32
EMBEDDING_SIZE = 128

model = M.RNNModel('RNN_TANH', VOCAB_SIZE, EMBEDDING_SIZE,
                   nhid=512, dropout=0.5)
if USE_CUDA:
    model = model.cuda()
model.load_state_dict(torch.load('lm-last.th'))
model.eval()


lines, words = utils.read_data('test_en.txt')
ds_test = M.Dataset(word2id, id2word, lines)
dl_test = DataLoader(ds_test)


with torch.no_grad():
    hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
    # 将数据按batch输入
    for i, batch in enumerate(dl_test):
        data, target = batch
        if USE_CUDA:
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
