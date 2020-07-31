#### config文件解读

##### 以inews_config.json为例

* model_name：模型名称
* epochs：迭代epoch的数量
* checkpoint_every：间隔多少步保存一次模型
* eval_every：间隔多少步验证一次模型
* learning_rate：学习速率，推荐2e-5， 5e-5， 1e-4
* sequence_length：序列长度，单GPU时不要超过128
* batch_size：单GPU时不要超过32
* num_classes：文本分类的类别数量，若是二分类设置为1
* warmup_rate：训练时的预热比例，建议0.05， 0.1
* output_path：输出文件夹，用来存储label_to_index等文件
* bert_model_path：预训练模型文件夹路径
* train_data：训练数据路径
* eval_data：验证数据路径
* ckpt_model_path：checkpoint模型文件保存路径


######处理好的数据
data文件夹下
训练数据中，train:valid=4:1


#########训练和测试
cd到classifier文件夹目录下
训练：sh run.sh
测试：CUDA_VISIBLE_DEVICES=2 python testnew.py

######运行环境
tensorflow-GPU-1.14-cp37（与代码不是最合适的版本，会有许多警告）