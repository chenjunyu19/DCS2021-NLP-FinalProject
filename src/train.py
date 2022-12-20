import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model as M
import utils

CONFIG = utils.read_config()

# 设置随机种子
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if CONFIG['useCUDA']:
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.cuda.set_device(1)

# 读取数据集
lines, words = utils.read_data('train_en.txt', CONFIG['tokenizer'])
lines_e, words_e = utils.read_data('eval_en.txt', CONFIG['tokenizer'])
# 生成 词语-id 映射
word2id, id2word = utils.make_map(words)

# 保存映射
if not os.path.exists(CONFIG['dataDir']):
    os.mkdir(CONFIG['dataDir'])
with open(os.path.join(CONFIG['dataDir'], 'word2id.json'), 'w') as f:
    json.dump(word2id, f)
with open(os.path.join(CONFIG['dataDir'], 'id2word.json'), 'w') as f:
    json.dump(id2word, f)

# 创建数据集
ds_train = M.Dataset(word2id, id2word, lines)
dl_train = DataLoader(ds_train)
ds_eval = M.Dataset(word2id, id2word, lines_e)
dl_eval = DataLoader(ds_eval)

# 创建 RNN 模型
VOCAB_SIZE = len(word2id)
LOSS_FN = nn.CrossEntropyLoss()
model = M.RNNModel('RNN_TANH', VOCAB_SIZE, CONFIG['embeddingSize'],
                   nhid=512, dropout=0.5)
if CONFIG['useCUDA']:
    model = model.cuda()


# critical：模型评估，建议先跳过这部分最后再看
def evaluate(model, dataloader):
    # 进入评估状态
    model.eval()
    total_loss = 0
    total_count = 0

    # 不是训练，关闭梯度加快运行速度
    with torch.no_grad():
        hidden = model.init_hidden(CONFIG['batchSize'], requires_grad=False)
        # 将数据按batch输入
        for i, batch in enumerate(dataloader):
            data, target = batch
            if CONFIG['useCUDA']:
                data, target = data.cuda(), target.cuda()

            hidden = M.repackage_hidden(hidden)

            with torch.no_grad():
                # model(data,hidden) 相当于调用model.forward
                output, hidden = model(data, hidden)

            # 计算损失
            loss = LOSS_FN(output.view(-1, VOCAB_SIZE), target.view(-1))

            total_count += np.multiply(*data.size())

            total_loss += loss.item() * np.multiply(*data.size())

        loss = total_loss / total_count
        model.train()

        return loss


optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learningRate'])

# critical：训练的主要流程
val_losses = []
losses = []
progressive = tqdm(range(CONFIG['numEpochs']))
for epoch in progressive:
    # critical：训练前务必手动设置model.train()
    # model.train()对应另一个函数model.eval(),前者有梯度用于训练，后者无梯度节省内存用于测试
    # 神经网络的后向传播和更新都依赖于梯度，没有梯度跑几个epoch都是无济于事
    model.train()
    hidden = model.init_hidden(CONFIG['batchSize'])

    # critical：将数据集中的数据按batch_size划分好，一一读入模型中
    for i, batch in enumerate(dl_train):
        data, target = batch

        # 使用gpu训练需要将数据也迁移到gpu
        if CONFIG['useCUDA']:
            data, target = data.cuda(), target.cuda()

        hidden = M.repackage_hidden(hidden)
        # critical：每步运行之前清空前一步backward留下的梯度，否则梯度信息不准确
        model.zero_grad()

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
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradClip'])

        # critical：optimizer更新模型参数
        optimizer.step()

    val_loss = evaluate(model, dl_eval)
    progressive.set_description(
        f'epoch: {epoch}, loss: {loss.item():.6f}, val_loss: {val_loss:.6f}')
    # critical：根据evaluate的结果，保存最好的模型
    if len(val_losses) == 0 or val_loss < min(val_losses):
        progressive.write(f'epoch {epoch} is the new best model')
        torch.save(model.state_dict(), os.path.join(
            CONFIG['dataDir'], 'state_dict_best.th'))
    val_losses.append(val_loss)
    losses.append({'loss': loss.item(), 'val_loss': val_loss})

    # 保存 checkpoint
    if (epoch + 1) % (CONFIG['numEpochs'] // 10) == 0:
        torch.save(model.state_dict(), os.path.join(
            CONFIG['dataDir'], f'state_dict_{epoch}.th'))

torch.save(model.state_dict(), os.path.join(
    CONFIG['dataDir'], 'state_dict_last.th'))
with open(os.path.join(CONFIG['dataDir'], 'losses.json'), 'w') as f:
    json.dump(losses, f)
