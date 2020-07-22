### ALBERT在下游任务中的应用

#### 本项目提供了易用的训练模式和预测模式，可以直接部署。也容易扩展到任何下游任务中

#### albert_task和bert_task文件夹中的内容基本一致

* albert_task/albert是albert的源码
* albert_task/albert_model中有albert_base ,可以包含四种albert的模型：albert_tiny, albert_base, albert_large, albert_xlarge进行训练，四种模型可见albert的readme.md文件夹
* 需要下载albert的预训练模型放置在albert_task\albert_model\albert_base下
* 预训练模型的路径可以在xxx_config.json文件中配置

#### sentence_pair_task

* config：放置每个具体任务的配置文件，包括训练参数，数据路径，模型存储路径
* data_helper.py：数据预处理文件
* metrics.py：性能指标文件
* model.py：模型文件，可以很容易的实现bert和下游网络层的结合
* trainer.py：训练模型
* test.py：预测代码，调用predict方法就可以预测

#### 训练数据格式

##### 句子对数据格式

* sentence A	sentence B	label：同样对于两个句子和标签采用'\t'符号分隔。
* 执行preprocess.py进行预处理

#### 训练模型

* 执行每个任务下的sh脚本即可，sh run.sh。只需要更改配置文件就可以训练不同的模型
*  CUDA_DEVICE_ORDER="PCI_BUS_ID_"CUDA_VISIBLE_DEVICES=1 sh /home/wangkehan/bert-for-task/albert_task/sentence_pair_task/run.sh
* 可能需要改变一下lcqmc_config.json中的文件路径

#### 预测

* 执行albert_task中sentence_pair_task的test.py文件就可以预测

  CUDA_DEVICE_ORDER="PCI_BUS_ID_"CUDA_VISIBLE_DEVICES=1  CUDA_VISIBLE_DEVICES=1 python /home/wangkehan/bert-for-task/albert_task/sentence_pair_task/test.py

  可能需要改变一下lcqmc_config.json中的文件路径