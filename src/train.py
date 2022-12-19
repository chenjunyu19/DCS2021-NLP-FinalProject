import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model as M
import utils

USE_CUDA = torch.cuda.is_available()

# 设置随机种子
SEED = 202212
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)
    torch.cuda.set_device(1)

# 超参设置
BATCH_SIZE = 32
EMBEDDING_SIZE = 128
# MAX_VOCAB_SIZE = 50000
GRAD_CLIP = 1
NUM_EPOCHS = 100

LOSS_FN = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001


lines, words = utils.read_data('train_en.txt')
word2id, id2word = utils.make_map(words)
VOCAB_SIZE = len(word2id)
with open('word2id.json', 'w') as f:
    json.dump(word2id, f)
with open('id2word.json', 'w') as f:
    json.dump(id2word, f)
ds_train = M.Dataset(word2id, id2word, lines)
dl_train = DataLoader(ds_train)

lines_e, words_e = utils.read_data('eval_en.txt')
ds_eval = M.Dataset(word2id, id2word, lines_e)
dl_eval = DataLoader(ds_eval)

model = M.RNNModel('RNN_TANH', VOCAB_SIZE, EMBEDDING_SIZE,
                   nhid=512, dropout=0.5)
if USE_CUDA:
    model = model.cuda()


# critical：模型评估，建议先跳过这部分最后再看
def evaluate(model, dataloader):
    # 进入评估状态
    model.eval()
    total_loss = 0
    total_count = 0

    # 不是训练，关闭梯度加快运行速度
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        # 将数据按batch输入
        for i, batch in enumerate(dataloader):
            data, target = batch
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            hidden = M.repackage_hidden(hidden)

            with torch.no_grad():
                output, hidden = model(data, hidden)

                # model(data,hidden) 相当于调用model.forward

            # 计算损失
            loss = LOSS_FN(output.view(-1, VOCAB_SIZE), target.view(-1))

            total_count += np.multiply(*data.size())

            total_loss += loss.item() * np.multiply(*data.size())

        loss = total_loss / total_count
        model.train()

        return loss


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# critical：训练的主要流程
val_losses = []
progressive = tqdm(range(NUM_EPOCHS))
for epoch in progressive:
    # critical：训练前务必手动设置model.train()
    # model.train()对应另一个函数model.eval(),前者有梯度用于训练，后者无梯度节省内存用于测试
    # 神经网络的后向传播和更新都依赖于梯度，没有梯度跑几个epoch都是无济于事
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)

    # critical：将数据集中的数据按batch_size划分好，一一读入模型中
    for i, batch in enumerate(dl_train):
        data, target = batch

        # 使用gpu训练需要将数据也迁移到gpu
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        hidden = M.repackage_hidden(hidden)
        model.zero_grad()  # critical：每步运行之前清空前一步backward留下的梯度，否则梯度信息不准确

        # print(data.size(), hidden[0].size())
        # critical：模型的forward，将数据正式传入模型中计算并输出结果
        # 输入：hidden：[BATCH_SIZE, seq_max_len]
        output, hidden = model(data, hidden)

        # critical：计算模型输出与真实标签的差距，也就是损失loss
        # 需要注意，设计模型时没有必要对output进行手动softmax为概率分布
        # nn.CrossEntropyLoss()会自动帮你完成这一步，否则二次softmax将导致模型训练不如预期
        loss = LOSS_FN(output.view(-1, VOCAB_SIZE), target.view(-1))

        # critical：梯度回传，准备更新模型参数
        loss.backward()

        # 解决梯度爆炸的问题
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # critical：optimizer更新模型参数
        optimizer.step()

    progressive.set_description(f'epoch: {epoch}, loss: {loss.item():.6f}')

    # 定时evaluate模型，查看模型训练情况
    if (epoch+1) % 10 == 0:
        progressive.write(f'[train] epoch: {epoch}, loss: {loss.item()}')
        val_loss = evaluate(model, dl_eval)
        progressive.write(f'[ val ] epoch: {epoch}, val_loss: {val_loss}')

        # critical：根据evaluate的结果，保存最好的模型
        if len(val_losses) == 0 or val_loss < min(val_losses):
            progressive.write(f'epoch {epoch} is the new best model')
            # critical：使用torch.save()保存模型到路径lm-best.th
            # 之后可以通过torch.load()读取保存好的模型
            torch.save(model.state_dict(), 'lm-best.th')

        val_losses.append(val_loss)

torch.save(model.state_dict(), 'lm-last.th')
