#!/bin/bash
python run.py --model TextCNN
python run.py --model TextCNN --embedding random
python run.py --model TextRNN
python run.py --model TextRNN --embedding random
python run.py --model TextRNN_Att
python run.py --model TextRNN_Att --embedding random
python run.py --model TextRCNN
python run.py --model TextRCNN --embedding random
python run.py --model Transformer
python run.py --model Transformer --embedding random