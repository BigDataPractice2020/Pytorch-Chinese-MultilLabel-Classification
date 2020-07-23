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
- [ ] TinyBert
- [x] FastBert
- [ ] BertToSimple

## 结果

## 参考文献

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
3. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
4. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
5. [FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://arxiv.org/abs/2004.02178)
6. [Transformer to CNN: Label-scarce distillation for efficient text classification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.03508)
7. [Distilling task-specific knowledge from bert into simple neural networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1903.12136)
8. [Distilling Transformers into Simple Neural Networks with Unlabeled Transfer Data](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.01769) 

