# Pytorch-Chinese-MultilLabel-Classification
Codes for the project

![](https://gitee.com/feiyuxiao/blogimage/raw/master/img/bairong.gif) 

## 目录

-[Requirements](#Requirements)

-[技术路线](#技术路线)
    [模型](#模型) 

-[结果](#结果)

-[参考文献](#参考文献) 

## Requirements

## 技术路线 

```
/Basic   作为 baseline 的 bert 和 Albert 的实现
/DistillBert 对于 Basic 中模型的蒸馏
/Preprocessor 前处理
/Postprocessor 后处理
```

### 模型

- [x] Roberta_wwm
- [x] Albert_zn
- [x] TinyBert
- [x] FastBert
- [x] BiGRU

## 模型说明
### 基础架构
![](https://gitee.com/feiyuxiao/blogimage/raw/master/img/bert_base.png)
特征向量生成：由原始特征经过特征统计生成
输入层：其中 X 表示节点类别，P 表示回答的内容，将两者进行拼接作为最终输入。
Transformer编码层：按照 Bert 和 albert 会有不同
输出层：一层全连接网络得到打分输出分类结果。

### 文本特征挖掘
![](https://gitee.com/feiyuxiao/blogimage/raw/master/img/pre_bert.png)
将每一类的类别信息词和文本信息进行拼接后再进行分类，如 [类别词][SEP][原始文本]

### BiGRU模型架构
![](https://gitee.com/feiyuxiao/blogimage/raw/master/img/bert_lstm.png)
词的embedding层是直接从teacher model上复制过来的，加快网络收敛速度。网络主干部分采用双向GRU。输出层部分是一个两层全连接网络，输入可以是GRU的输出也可以是GRU最后一个时间步的隐藏单元，分别对应Model A 和B。采用Model B时，因为输入词长短不一，为了得到真正的最后一个字输出的隐藏单元，需要利用for循环进行单时间步的训练，所以速度会慢一些。Model A 和B准确率都能达到90%。损失函数时MSE和交叉熵的结合，分别拟合teacher model的logits 和真正的标签信息。

## 效果对比
![](https://gitee.com/feiyuxiao/blogimage/raw/master/img/bert_final.png)


## 参考文献

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
3. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
4. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
5. [FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://arxiv.org/abs/2004.02178)
6. [Transformer to CNN: Label-scarce distillation for efficient text classification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.03508)
7. [Distilling task-specific knowledge from bert into simple neural networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1903.12136)
8. [Distilling Transformers into Simple Neural Networks with Unlabeled Transfer Data](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.01769) 

