from typing import List

import torch
import torch.nn as nn


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, word2id: dict, id2word: dict, texts: List[List[str]]) -> None:
        super().__init__()
        self.word2id = word2id
        self.id2word = id2word
        self.texts = [self.get_ids_by_words(text) for text in texts]

    def get_words_by_ids(self, ids: List[int]) -> str:
        return ' '.join([self.id2word[id] for id in ids])

    def get_ids_by_words(self, words: List[str]) -> List[int]:
        # 将未登陆词标记为 <unk>
        return [self.word2id[word if word in self.word2id else '<unk>'] for word in words]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sent = self.texts[idx]
        target = sent[1:] + [0]
        sent = torch.tensor(sent)
        target = torch.tensor(target)
        return sent, target


class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, dropout=0.5):
        super(RNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.vocab_size = ntoken
        self.nhid = nhid

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("unknown parameter, you can use \
                    ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']")
            self.rnn = nn.RNN(ninp, nhid, nonlinearity=nonlinearity,
                              dropout=dropout, batch_first=True)

    # critical：定义前向传播
    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        # output：每一次输入的输出
        # hidden：最后一次的输出，会被decoder直接使用
        output, hidden = self.rnn(emb)
        output = self.drop(output)
        # critical：然后进行线性转换，view函数的用处是改变tensor的形状
        # 因为我们要训练，所以对每个位置的字词都要预测一次，所以使用包含所有字词的output
        decoded = self.decoder(output.view(
            output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((1, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((1, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((1, bsz, self.nhid), requires_grad=requires_grad)
