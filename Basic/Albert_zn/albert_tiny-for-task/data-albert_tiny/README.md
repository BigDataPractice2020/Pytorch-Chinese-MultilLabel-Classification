1、数据集格式说明：
文本分类数据格式：title <SEP> content <SEP> label。title使用类别标签，即type_robot；content使用msg对应的文本

参考：https://github.com/jiangxinyang227/bert-for-task

2、数据划分说明
项目方提供的7个单独的训练集混合并打乱顺序后划分为训练集和验证集，比例自定（当前目录的train：valid=4:1）。
测试集直接将7个测试集按顺序混合。

3、标签说明
identity1_train：0(invalid),1(yes),2(no)
once1_train：3,4,5
ask_know_train：6,7,8
twice_train：9,10,11
request_train：12,13,14
ask_today_train：15,16,17
ask_tomorrow_train：18,19,20
