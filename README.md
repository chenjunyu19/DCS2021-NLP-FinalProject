# DCS2021-NLP-FinalProject

这是 2022 年秋季学期公选课 `DCS2021-自然语言处理导论` 的期末大作业。

## 外部文件来源

出于便利考虑，此仓库中包含部分外部来源文件。

- `{train,eval,test}_en.txt`：数据集，来自[作业示例项目](https://github.com/djz233/DCS2021)。

## 开发环境

```
x86_64 GNU/Linux Ubuntu
Python 3.8.15

numpy
torch==1.13.0
torchtext==0.14.0
tqdm
```

## 使用说明

将仓库克隆到本地、处理好依赖关系后，在仓库根目录打开终端。

执行 `train.py` 进行训练，会均匀挑选训练过程中约 10 个时刻保存模型。如果已经有保存好的模型想要加载，则不需要训练，将相关文件放好后执行预测程序即可。

```
python src/train.py
```

执行 `predict.py` 进行预测。等待程序提示输入后，可以输入想要加载的模型，对测试集（`test_en.txt`）中的句子分别预测下一个单词。

`best` 为验证集上损失最小的模型，`last` 为训练结束时的模型，还可以查看 `checkpoints` 文件夹后输入整数选择特定时刻的模型。如果没有改动任何配置，在此数据集应当使用 `last`。

```
python src/predict.py
```

其他文件：

- `utils.py`：数据预处理（文件读取、分词）
- `model.py`：模型实现
- `plot.py`：写报告画图用的
