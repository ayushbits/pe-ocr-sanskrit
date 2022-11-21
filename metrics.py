import pandas as pd
import editdistance as ed
from collections import defaultdict
import argparse
import sys
import os
def calculate_metrics(predicted_text, transcript):
    cer = ed.eval(predicted_text, transcript) / max(len(predicted_text), len(transcript))
    pred_spl = predicted_text.split()
    transcript_spl = transcript.split()
    wer = ed.eval(pred_spl, transcript_spl) /  max(len(pred_spl), len(transcript_spl))
    return cer, wer


def rem(s):
  # print(s)
  return s.replace("\n",'')


def get_metric(filename):
  df_output = pd.read_csv(filename, sep=',')
  df_output['input_text'] = df_output['input_text'].apply(lambda x: rem(x))
  df_output['target_text'] = df_output['target_text'].apply(lambda x: rem(x))
  df_output['predicted_text'] = df_output['predicted_text'].apply(lambda x: rem(x))

  df_output['input_text'] = df_output['input_text'].apply(lambda x:x.rstrip().lstrip())
  df_output['target_text'] = df_output['target_text'].apply(lambda x:x.rstrip().lstrip())
  df_output['predicted_text'] = df_output['predicted_text'].apply(lambda x:x.rstrip().lstrip())

  for index, row in df_output.iterrows():
    ref = row['target_text']
    output = row['predicted_text']
    # output = row['input_text']
    
    cer,wer = calculate_metrics(output,ref)
    df_output.loc[df_output['img_filename'] == filename, 'cer'] = round(cer,2) # Round value to 2 decimal places
    df_output.loc[df_output['img_filename'] == filename, 'wer'] = round(wer,2)

  # Overall performances
  mean_cer = df_output['cer'].mean()
  mean_wer = df_output['wer'].mean()
  print(f'Mean CER = {mean_cer}%, Mean WER = {mean_wer}%')
  return mean_cer, mean_wer

if __name__=="__main__":

  fname = sys.argv[1]
  cer, wer = get_metric(sys.argv[1])
