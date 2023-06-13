# 22Summer-KDD

1. 运行sr_san_preprocess.py

   1. 创建`data`文件夹，在data文件夹中分别建立`UK、DE、JP`三个文件夹

      文件目录如图：

      <img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230613131813178.png" alt="image-20230613131813178" style="zoom: 67%;" />

   2. 交替进行。

      `get_counts()`对每种商品的评价数量计数

      `get_embeddings()`对标题和`desc`（是什么啊）进行encoding，使用sentence_transformers。

      > ​	Running on RTX3060 12G LHR, 170/170W, used/all 9/12G.
      >
      > ​	Around 1.5h.

      <img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230613132718083.png" alt="image-20230613132718083" style="zoom:67%;" />

      

