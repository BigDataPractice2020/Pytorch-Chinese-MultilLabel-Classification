""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
# from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification # chinese version
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar
from processors.glue import MrpcProcessor
def train(args, train_dataset_list, model, tokenizer):
    """ Train the model """
    # 把这个变成1 不然之后会数据并行
    # args.n_gpu = 1
    args.train_batch_size = args.per_gpu_train_batch_size
    train_dataloader_list = []
    # 一共有多少个数据
    all_data_num = 0
    for i in range(7):
        train_sampler = RandomSampler(train_dataset_list[i])
        train_dataloader_list.append(DataLoader(train_dataset_list[i], sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn))
        all_data_num += len(train_dataset_list[i])

    # 一共训练了多少条数据
    num_training_steps = all_data_num // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", all_data_num)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        # 显示进度的一个对象
        pbar = ProgressBar(n_total=all_data_num//args.per_gpu_train_batch_size, desc='Training')
        # 将数据按batch取出缓存起来
        all_link_batch_data_cache = []
        all_link_batch_num = []
        for i in range(7):
            # 内容为一个元组，元组第一个位置为属于哪个link，第二个位置是batch_data
            for batch in train_dataloader_list[i]:
                all_link_batch_data_cache.append((i,batch))
        all_link_batch_data_cache = np.array(all_link_batch_data_cache)
        #打乱顺序
        np.random.shuffle(all_link_batch_data_cache)
        step = 0
        for linknum_batch in all_link_batch_data_cache:
            step += 1
            model.train()
            link_num = linknum_batch[0]
            batch = linknum_batch[1]
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      "num_link":link_num,
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            # print(inputs)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            # 还能这么玩？ 多个step才更新一次梯度
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            #每隔每隔logging stepeval一次
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #Log metrics
                # 1个gpu必须eval
                if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                    evaluate(args, model, tokenizer)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
            pbar(step, {'loss': loss.item()})
        print(" ")
        if 'cuda' in str(args.device):
            del batch,inputs,outputs
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset_list = []
        for i in range(7):
            link_name = "link{}".format(i)
            eval_dataset_list.append(load_and_cache_examples(args, eval_task, tokenizer, link_name, data_type='dev'))
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        all_data_num = 0
        eval_dataloader_list = []
        for i in range(7):
            eval_sampler = SequentialSampler(eval_dataset_list[i])
            eval_dataloader_list.append(DataLoader(eval_dataset_list[i], sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn))
            all_data_num += len(eval_dataset_list[i])

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", all_data_num)
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=all_data_num//args.eval_batch_size, desc="Evaluating")

        all_link_batch_data_cache = []
        for i in range(7):
            # 内容为一个元组，元组第一个位置为属于哪个link，第二个位置是batch_data
            for batch in eval_dataloader_list[i]:
                all_link_batch_data_cache.append((i, batch))
        all_link_batch_data_cache = np.array(all_link_batch_data_cache)
        step = 0
        for linknum_batch in all_link_batch_data_cache:
            step+=1
            model.eval()
            link_num = linknum_batch[0]
            batch = linknum_batch[1]
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          "num_link":link_num,
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
            # if step == 10:
            #     break

        print(' ')
        if 'cuda' in str(args.device):
            del batch,inputs,outputs
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        # 这里重新定义了mrpc任务，是多分类的
        if args.output_mode == "classification" and eval_task == "mrpc":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        #不同的任务是不同的指标，mprc任务返回的是acc和f1 score
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return results

# 直接返回dataloader
def load_and_cache_examples(args, task, tokenizer,link_name,data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor =MrpcProcessor()
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir,link_name,'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # 看看有没有缓存起来的处理过格式的数据
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", os.path.join(args.data_dir,link_name))
        label_list = processor.get_labels()

        if data_type == 'train':
            examples = processor.get_train_examples(os.path.join(args.data_dir,link_name))
            # examples是一个列表，列表里边是很多个字典包括:序号，句子1，句子2，是否匹配(实际是json)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(os.path.join(args.data_dir,link_name))
        else:
            print("no test data")
            raise NameError
            examples = processor.get_test_examples(os.path.join(args.data_dir,link_name))
        # 将句子分字并用数字代替，不足max_len的用0pad  token_type_ids只有0，1两个数值，标识是第一句还是第二句
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode = output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./dataset/cuishou/", type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # 模型类型 貌似只有roberta才会做一些额外处理
    parser.add_argument("--model_type", default="albert", type=str, required=False,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default="./prev_trained_model/albert_large_zh/", type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list")
    # 八种glue任务之一
    parser.add_argument("--task_name", default="MRPC", type=str, required=False,
                        help="task type")
    # 输出路径
    parser.add_argument("--output_dir", default="./outputs", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # 词表地址
    parser.add_argument("--vocab_file",default='./prev_trained_model/albert_large_zh/vocab.txt', type=str)
    # 这个也是词表？直接填NONE
    parser.add_argument("--spm_model_file",default=None,type=str)

    ## Other parameters
    # 空着就行，如果没有会自动找model_name_or_path的
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", default=True,type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False,type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    # 每多少个batch评估一次eval
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    # 每多少batch保存一次模型
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=True,type=bool,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    #是否有输出文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # 设置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Setup logging
    logger.warning("Process rank: %s, device: %s",args.local_rank, device)
    # Set seed
    seed_everything(args.seed)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    # processors字典里边存的是函数的指针，根据任务类别的字符串索引，并构建对应的类对象
    processor = MrpcProcessor()
    print(args.task_name)
    # output_mode指的是他是分类任务还是回归任务
    args.output_mode = "classification"
    # 一共有7种环节
    link_nums = 7
    # 得到标签列表 如二分类0，1
    label_list = processor.get_labels()
    print(label_list)
    num_labels = len(label_list)
    # albert
    args.model_type = args.model_type.lower()
    print(args.config_name)
    # 返回一个字典，包含模型目录下config文件里的内容+分类数+任务名称
    config = AlbertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    # 一个能够处理文字格式的对象，包括预处理，文字->数字的映射
    tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                 spm_model_file=args.spm_model_file)
    # 保存的预训练模型没有全链接decoder，from_pretrained加载了encoder自动在后边加一个decoder
    # from_pretrained是AlbertForSequenceClassification父类的一个classmethod函数
    # 先执行from_pretrained，里边CLS()有实例化模型的代码，再执行__init__
    model =AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                           from_tf=bool('.ckpt' in args.model_name_or_path),
                                                            config=config)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # 加载7份训练集
        train_dataset_list = []
        for i in range(link_nums):
            link_name = "link{}".format(i)
            train_dataset_list.append(load_and_cache_examples(args, args.task_name, tokenizer,link_name,data_type='train'))

        global_step, tr_loss = train(args, train_dataset_list, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        #最优模型保存在output_dir
        model_to_save.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file,
                                                      do_lower_case=args.do_lower_case,
                                                      spm_model_file=args.spm_model_file)
        checkpoints = [(0,args.output_dir)]
        #每个epoch会保存一个训练模型，在这对每个保存的模型全都测试一下
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]),checkpoint) for checkpoint in checkpoints if checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints,key =lambda x:x[0])
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _,checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model =AlbertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key,value in results:
                writer.write("%s = %s\n" % (key, str(value)))

if __name__ == "__main__":
    main()
