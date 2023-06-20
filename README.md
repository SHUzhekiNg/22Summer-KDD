# Overview

product_train.csv

| id    | locale   | title | price | brand | color | size | model | material | author | desc |
| ----- | -------- | ----- | ----- | ----- | ----- | ---- | ----- | -------- | ------ | ---- |
| **T** | DE/JP/UK | **T** |       | T     | T     | T    | T     |          |        |      |



session_train.csv

| prev_items | next_item | locale   |
| ---------- | --------- | -------- |
| id         | id        | DE/JP/UK |



1、只基于ID序列的下一个物品推荐：难度系数0.8

==2、基于ID序列和单语言的下一个物品推荐：难度系数1.0==

3、基于ID序列和多语言的下一个物品推荐：难度系数1.2

4、基于ID序列和单语言的下一个标题预测：难度系数1.2

5、基于ID序列和多语言的下一个标题预测：难度系数1.4



# Pipeline

1. Make directory `data` under the project root. For instance,
   
   
   ![image-20230620132011366](https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230620132011366.png)
2. run command `pip install -r requirements.txt`
   Tested under Python3.9/3.10, CUDA 11.7
   CUDNN8 are required for Tensorflow.

### Preprocessing

`cd preprocessing/`

1. run `preprocessing.py`
   *may include warnings but will not impact the overall performance.*
2. run `processing_nvt.py`

### Training

`cd model/`

- For Tensorflow, run `train.py`
- For Pytorch, run `train_torch.py`
