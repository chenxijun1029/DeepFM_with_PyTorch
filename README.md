# DeepFM_with_PyTorch

A PyTorch implementation of DeepFM for CTR prediction problem.

## Usage

1. Download Criteo's Kaggle display advertising challenge dataset from [here][1]( if you have had it already, skip it ), and put it in *./data/raw/*

2. Generate a preprocessed dataset.

        ./utils/dataPreprocess.py

3. Train a model and predict.

        ./main.py

## Output


## Reference

- https://github.com/nzc/dnn_ctr.

- https://github.com/PaddlePaddle/models/tree/develop/deep_fm.

- DeepFM: A Factorization-Machine based Neural Network for CTR         Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.



[1]: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
