from transformers import HfArgumentParser, TensorFlowBenchmark, TensorFlowBenchmarkArguments
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
#import datasets



train_df = pd.read_csv("train_transliterated.csv")
eval_df = pd.read_csv("eval_transliterated.csv")
test_df = pd.read_csv("test_transliterated.csv")


args_dict = {
    "model_name_or_path": '../facebook/mbart-large-50_deploy',
    "max_len": 164,
    "output_dir": './bart-base-sanskrit-ocr-correction',
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "warmup_steps": 250,
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 250,
    "num_train_epochs": 4,
    "do_train": True,
    "do_eval": True,
    "fp16": False,
    "use_cache": False,
    "max_steps": 100000
}
parser = HfArgumentParser(
        (TrainingArguments))
training_args = parser.parse_dict(args_dict)
# set_seed(training_args.seed)
# set_seed(training_args.seed)
args = training_args[0]

# Load pretrained model and tokenizer

# Load pretrained model and tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt",
    cache_dir="content",
    max_length=164
)
model = MBartForConditionalGeneration.from_pretrained(
    "../facebook/mbart-large-50_deploy/",
    cache_dir=".",
)

# overwriting the default max_length of 20 
tokenizer.model_max_length=164
model.config.max_length=164

# overwriting the default max_length of 20 
# tokenizer.model_max_length=512
# model.config.max_length=512



class GPReviewDataset(Dataset):
  def __init__(self, Text, Label):
    self.Text = Text
    self.Label = Label
    # self.tokenizer = tokenizer
    # self.max_len = max_len
  def __len__(self):
    return len(self.Text)
  def __getitem__(self, item):
    Text = str(self.Text[item])
    Label = self.Label[item]
    inputs = tokenizer(Text, padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(Label, padding="max_length", truncation=True, max_length=512)
    return {
      "input_ids":inputs.input_ids,
      "attention_mask" : inputs.attention_mask,
      "labels" : outputs.input_ids,
      "decoder_attention_mask" : outputs.attention_mask,
      # "labels" : lbz
    }

ds_train = GPReviewDataset(
  Text=train_df.input_text.to_numpy(),
  Label=train_df.target_text.to_numpy()
  # tokenizer=tokenizer,
  # max_len=max_len
)


ds_test = GPReviewDataset(
  Text=eval_df.input_text.to_numpy(),
  Label=eval_df.target_text.to_numpy()
  # tokenizer=tokenizer,
  # max_len=max_len
)


train_dataset = ds_train
valid_dataset = ds_test


# load rouge for validation
#rouge = datasets.load_metric("rouge")

#def compute_metrics(pred):
#    labels_ids = pred.label_ids
#    pred_ids = pred.predictions

    # all unnecessary tokens are removed
#    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

#    return {
#        "rouge2_precision": round(rouge_output.precision, 4),
#        "rouge2_recall": round(rouge_output.recall, 4),
#        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#    }



trainer = Trainer(  
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics

)

trainer.args.save_total_limit = 2
trainer.train()#'bart-base-sanskrit-ocr-correction/checkpoint-12000')
