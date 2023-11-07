# 船舶路径分类
> 自用 数据集特殊

参考大佬：https://github.com/649453932/Chinese-Text-Classification-Pytorch

TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

## 介绍

模型介绍、数据流动过程：[我的博客](https://zhuanlan.zhihu.com/p/73176084)

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)


### 更换自己的数据集

- 如果用字，按照我数据集的格式来格式化你的数据。
- 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`
- 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。

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

bert和ERNIE模型代码我放到另外一个仓库了，传送门：[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)
，后续还会搞一些bert之后的东西，欢迎star。

### 参数

模型都在models目录下，超参定义和模型定义在同一文件中。

## 对应论文

[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[4] Recurrent Convolutional Neural Networks for Text Classification  
[5] Bag of Tricks for Efficient Text Classification  
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[7] Attention Is All You Need  
