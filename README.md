# 船舶路径分类

> 项目自用 数据集为船舶特殊数据集

参考大佬：https://github.com/649453932/Chinese-Text-Classification-Pytorch

TextCNN DPCNN

TextRNN TextRCNN BiLSTM_Attention

Transformer

BERT

基于pytorch，开箱即用。

## 介绍

模型介绍、数据流动过程：[我的博客](https://zhuanlan.zhihu.com/p/73176084)

## 效果

| 模型          | acc    | 备注                         |
|-------------|--------|----------------------------|
| TextCNN     | 91.22% | Kim 2014 经典的CNN文本分类        |
| TextRNN     | 91.12% | BiLSTM                     |
| TextRNN_Att | 90.90% | BiLSTM+Attention           |
| TextRCNN    | 91.54% | BiLSTM+池化                  |
| FastText    | 92.23% | bow+bigram+trigram， 效果出奇的好 |
| DPCNN       | 91.25% | 深层金字塔CNN                   |
| Transformer | 89.91% | 效果较差                       |
| bert        | 94.83% | bert + fc                  |
| ERNIE       | 94.61% | 比bert略差(说好的中文碾压bert呢)      |



## 对应论文

[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[4] Recurrent Convolutional Neural Networks for Text Classification  
[5] Bag of Tricks for Efficient Text Classification  
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[7] Attention Is All You Need  
