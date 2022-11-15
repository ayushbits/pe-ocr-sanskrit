from transformers import HfArgumentParser, TensorFlowBenchmark, TensorFlowBenchmarkArguments
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import Trainer
from transformers import pipeline
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small",
    cache_dir="content",
    max_length=512
)

test_df = pd.read_csv("final_trans_data_OCR/test.csv")

print('Test csv read')
ocr_pipeline = pipeline(
    'text2text-generation',
    model="bart-base-sanskrit-ocr-correction/checkpoint-63000",
    tokenizer=tokenizer,device = 2)

print('Model Loaded')
results=  [] 
data = list(test_df.input_text.values)[:10]
print(data)
#for i in list(test_df.input_text.values):
#    results.append(ocr_pipeline(i))
#results = ocr_pipeline(list(test_df.input_text.values))
results = ocr_pipeline(data)

pred_resultz = []
for i in tqdm(list(range(len(results)))):
  for k,e in results[i].items():
    pred_resultz.append(e)

res = pd.DataFrame(zip(test_df.input_text.values,test_df.target_text.values,pred_resultz),columns = ['input_text','target_text','predicted_text'])

res.to_csv("BART_predictions_63k.csv",index = False)
