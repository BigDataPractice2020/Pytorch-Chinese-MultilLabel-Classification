进行分类任务直接运行run_classifier.py
*********************************************************************
对main函数开始设置的参数进行说明：
model_name_or_path——teahcer model的位置，是分类正确率为93%的ALBERT
continue_train_distill——是否加载训练好（训练一半）的蒸馏模型的参数
distill_model_dir——训练好的蒸馏模型的参数位置，分类正确率为90%
do_train和do_eval——是否进行训练/推理
logging_steps——在训练时，训练多少批数据做一次在测试集上预测
save_steps——训练多少批数据保存一次模型
*********************************************************************
my_model.py保存了蒸馏模型网络结构


