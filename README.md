# Goal-Driven Tree-Structured Neural Models for Solving Math Word Problems

This repository is the [PyTorch](http://pytorch.org/) implementation for the IJCAI 2019 accepted paper:
> Zhipeng Xie* and Shichao Sun*,
> [Goal-Driven Tree-Structured Neural Models for Solving Math Word Problems](http://www.ke.fudan.edu.cn/papers/ijcai2019-math.pdf)
> IJCAI 2019. 

\* indicates equal contribution.

## Seq2Tree Model
A Seq2Tree Neural Network containing top-down Recursive Neural Network and bottom-up Recursive Neural Network

<img src='readme/tree_decoder.png' align="center" width="700px">


## Requirements
- python 3
- [PyTorch](http://pytorch.org/) 0.4.1


## Train and Test

- Math23K: 
```
python3 run_seq2tree.py
```

## Results

| Model | Accuracy | 
|--------|--------|
|Hybrid model w/ SNI | 64.7% | 
|Ensemble model w/ EN | 68.4% | 
|Seq2Tree w/o Bottom-up RvNN | 70.0% | 
|Seq2Tree| **74.3%** | 
