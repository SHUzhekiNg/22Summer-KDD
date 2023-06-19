

product_train.csv

| id   | locale | title | price | brand | color | size | model | material | author | desc |
| ---- | ------ | ----- | ----- | ----- | ----- | ---- | ----- | -------- | ------ | ---- |
| T    |        | T     |       |       |       |      |       |          |        |      |



session_train.csv

| prev_items | next_item | locale |
| ---------- | --------- | ------ |
|            |           |        |

1、只基于ID序列的下一个物品推荐：难度系数0.8

2、基于ID序列和单语言的下一个物品推荐：难度系数1.0

3、基于ID序列和多语言的下一个物品推荐：难度系数1.2

4、基于ID序列和单语言的下一个标题预测：难度系数1.2

5、基于ID序列和多语言的下一个标题预测：难度系数1.4



