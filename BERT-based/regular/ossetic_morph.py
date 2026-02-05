from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from torch.optim import AdamW
from torch.utils.data.dataset import Dataset
import numpy as np
from collections import Counter
import subprocess
import argparse
from huggingface_hub import login

parser = argparse.ArgumentParser()
#parser.add_argument('--output_dir', type=str, required=True)

parser.add_argument('--model_name', type=str, required=True)

parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument('--eval_steps', type=int, default = 200)
parser.add_argument('--save_steps', type=int, default = 200) 
parser.add_argument('--per_device_train_batch_size', type=int, default = 8)
parser.add_argument('--per_device_eval_batch_size', type=int, default = 8)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--max_steps", type=int)
group.add_argument("--num_train_epochs", type=int)

args = parser.parse_args()

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
urls = [
    "https://raw.githubusercontent.com/ania3000/Ossetic-COT/master/train.conllu",
    "https://raw.githubusercontent.com/ania3000/Ossetic-COT/master/dev.conllu",
    "https://raw.githubusercontent.com/ania3000/Ossetic-COT/master/test.conllu",
]
for url in urls:
    subprocess.run(["wget", url], check=True)

def make_last_subtoken_mask(mask, has_cls=True, has_eos=True):
    if has_cls:
        mask = mask[1:]
    if has_eos:
        mask = mask[:-1]
    is_last_word = list((first != second) for first, second in zip(mask[:-1], mask[1:])) + [True]
    if has_cls:
        is_last_word = [False] + is_last_word
    if has_eos:
        is_last_word.append(False)
    return is_last_word

def read_infile(infile):
    answer, sent, labels = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(sent) > 0:
                    answer.append({"words": sent, "labels": labels})
                sent, labels = [], []
                continue
            splitted = line.split("\t")
            if not splitted[0].isdigit():
                continue
            if len(splitted)<4:
                continue
            tag = splitted[3] if splitted[5] == "_" else f"{splitted[3]},{splitted[5]}"
            sent.append(splitted[1])
            labels.append(tag)
    if len(sent) > 0:
        answer.append({"words": sent, "labels": labels})
    return answer

train_data = read_infile("train.conllu")
dev_data = read_infile("dev.conllu")
test_data = read_infile("test.conllu")

class UDDataset(Dataset):

    def __init__(self, data, tokenizer, min_count=1, tags=None): 
        self.data = data
        self.tokenizer = tokenizer
        if tags is None:
            tag_counts = Counter([tag for elem in data for tag in elem["labels"]])
            self.tags_ = ["<PAD>", "<UNK>"] + [x for x, count in tag_counts.items() if count >= min_count]
        else:
            self.tags_ = tags
        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.unk_index = 1
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        answer = {"input_ids": tokenization["input_ids"], "mask": last_subtoken_mask}
        if "labels" in item:
            labels = [self.tag_indexes_.get(tag, self.unk_index) for tag in item["labels"]]
            zero_labels = np.array([self.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
            answer["labels"] = zero_labels
        return answer
    
train_dataset = UDDataset(train_data, tokenizer)
dev_dataset = UDDataset(dev_data, tokenizer, tags=train_dataset.tags_)
test_dataset = UDDataset(test_data, tokenizer, tags=train_dataset.tags_)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_labels = np.argmax(logits, axis=-1)
    correct, total, seq_correct = 0, 0, 0
    for i, (pred_sent_labels, sent_labels) in enumerate(zip(pred_labels, labels)):
        is_correct = True
        for pred_label, label in zip(pred_sent_labels, sent_labels):
            if label != -100:
                if pred_label == label:
                    correct += 1
                else:
                    is_correct = False
                total += 1
        seq_correct += int(is_correct)
    return {"Accuracy": 100 * correct / total, "Sentence accuracy": 100 * seq_correct / len(labels)}

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_dataset.tags_))

training_args_kwargs = dict(
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    save_strategy="steps",
    eval_strategy="steps",
    logging_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    logging_steps = args.eval_steps,
    disable_tqdm=False,
    learning_rate=args.learning_rate,
    report_to="none",
    weight_decay=0.01,
    #output_dir=args.output_dir
)

if args.max_steps is not None:
    training_args_kwargs["max_steps"] = args.max_steps
elif args.num_train_epochs is not None:
    training_args_kwargs["num_train_epochs"] = args.num_train_epochs

training_args = TrainingArguments(**training_args_kwargs)

trainer = Trainer(
    model=model,
    optimizers=(AdamW(model.parameters(), lr=5e-5, weight_decay=0.01), None),
    args=training_args,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics)

trainer.train()
predictions = trainer.predict(test_dataset)
print(predictions.metrics["test_Accuracy"])

print(predictions.metrics["test_Sentence accuracy"])


