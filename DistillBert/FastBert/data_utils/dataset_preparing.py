import torch
import data_utils.tokenization as tokenization
import csv

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas

def text_collate_fn(batch, padding):
    texts = [instance["text"] for instance in batch]
    tokens = [instance["tokens"] for instance in batch]
    segment_ids = [[0] * len(token) for token in tokens]
    attn_masks = [[1] * len(token) for token in tokens]
    labels = [instance["label"] for instance in batch]
    max_len = max([len(token) for token in tokens])
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    labels = torch.LongTensor(labels)
    return {"texts":texts, "tokens" : tokens, "segment_ids" : segment_ids,
            "attn_masks" : attn_masks, "labels" : labels}


class TextCollate():
    def __init__(self, dataset, tag_padding=0):
        self.padding = dataset.tokenizer.vocab["[PAD]"]

    def __call__(self, batch):
        return text_collate_fn(batch, self.padding)

class PrepareDataset(torch.utils.data.Dataset):

    def __init__(self, num_class, max_seq_len, data_file=None, vocab_file=None, do_lower_case=True):
        self.max_seq_len = max_seq_len
        self.num_class = num_class

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

        if data_file != None:
            self.raw_data = open(data_file, 'r').readlines()


    def __len__(self):
        return len(self.raw_data)


    '''
    def __getitem__(self, index):
        line = self.raw_data[index]
        row = line.strip("\n").split("\t")
        if len(row) != 2:
            raise RuntimeError("Data is illegal: " + line)

        # __label__0, __label__1
        if len(row[0]) == 10:
            label = int(row[0][-1])
        # 0, 1
        else:
            label = int(row[0])
        if label > self.num_class - 1:
            raise RuntimeError("data label is illegal: " + line)

        tokens = self.tokenizer.tokenize(row[1])
        tokens = tokens[:(self.max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return {"text":row[1], "tokens" : tokens, "label" : label}
    '''
    def __getitem__(self, index):
        line = self.raw_data[index]
        row = line.strip("\n").split("\t")
        if len(row) != 2:
            raise RuntimeError("Data is illegal: " + line)

        # __label__0, __label__1
        if len(row[1]) == 10:
            label = int(row[1][-1])
        # 0, 1
        else:
            label = int(row[1])
        if label > self.num_class - 1:
            raise RuntimeError("data label is illegal: " + line)

        tokens = self.tokenizer.tokenize(row[0])
        tokens = tokens[:(self.max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return {"text":row[0], "tokens" : tokens, "label" : label}
            

