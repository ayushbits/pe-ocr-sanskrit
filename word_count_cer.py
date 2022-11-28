import fastwer
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import sys

mode = sys.argv[1]
if mode == 'cer':
    mode = True
elif mode =='wer':
    mode = False
else:
    print('argv[1] can be either cer or wer')
    exit()

df_byt5 = pd.read_csv('preds/byt5_500.csv',sep=';')
df_mt5 = pd.read_csv('preds/mt5_500.csv',sep=';')


word_cer, count ={}, {}
ocr_cer ,ocr_count = {}, {}
# print(df_byt5.head())
for idx, row in df_byt5.iterrows():
    inp, tgt, pred = row['input_text'], row['target_text'], row['predicted_text']
    t = tgt.split()
    p = pred.split()
    o = inp.split()
    # if len(t) == len(p):
    for i, j, k in zip(t,p,o):
        # i, j = str(i), str(j)
        if len(i) not in word_cer.keys():
            word_cer[len(i)] = 0
            count[len(i)] = 0
        cer = fastwer.score_sent(j, i, char_level=mode)
        # if len(p) == len(o):
            
        ocer = fastwer.score_sent(k, i, char_level=mode)
        if ocer >= 0:
            if len(i) not in ocr_cer.keys():
                ocr_cer[len(i)] = 0
                ocr_count[len(i)] = 0
            
            ocr_cer[len(i)] += ocer  
            ocr_count[len(i)] += 1
        if cer >= 0:
            word_cer[len(i)] += cer  
            count[len(i)] += 1



pred_cer = {v: float(word_cer[v]/count[v]) for v in word_cer.keys()}
preds = sorted(pred_cer.items())
_, preds_l = zip(*preds)
# print('COunt of ByT5 ', sorted(pred_cer.items()))

ocr_cer = {v: float(ocr_cer[v]/ocr_count[v]) for v in ocr_cer.keys()}
ocrs = sorted(ocr_cer.items())
_, ocrs_l = zip(*ocrs)
# print('Count of OCR ', sorted(ocr_cer.items()))
# print('Count of ByT5 ', sorted(count.items()))

count = sorted(count.items())
count_l , _ = zip(*count)
# print('COunt OCR of Words ', sorted(ocr_count.items()))

ocr_count = sorted(ocr_count.items())
ocr_count_l , _ = zip(*ocr_count)

## Calculating CER for MT5
print('Average ByT5', np.average(preds_l))
print('Average OCR', np.average(ocrs_l))

## Calculating CER for MT5

# df_mt5 = pd.read_csv('output_byt5_noearly/mt5_Preds_Devnagri_external.csv',sep=';')


word_cer_mt5, count_mt5 ={}, {}
# print(df_mt5.head())
for idx, row in df_mt5.iterrows():
    inp, tgt, pred = row['input_text'], row['target_text'], row['predicted_text']
    t = tgt.split()
    p = pred.split()
    # o = inp.split()
    # if len(t) == len(p):
    for i, j in zip(t,p):

        # i, j = str(i), str(j)
        if len(i) not in word_cer_mt5.keys():
            word_cer_mt5[len(i)] = 0
            count_mt5[len(i)] = 0
            
        cer = fastwer.score_sent(j, i, char_level=mode)
        if cer >= 0:
            cer = cer
            word_cer_mt5[len(i)] += cer  
            count_mt5[len(i)] += 1

pred_cer_mt5 = {v: float(word_cer_mt5[v]/count_mt5[v]) for v in word_cer_mt5.keys()}
preds_mt5 = sorted(pred_cer_mt5.items())
_, preds_mt5_l = zip(*preds_mt5)

# print('CER_mt5 of Words ', sorted(pred_cer_mt5.items()))
# print('COunt_mt5 of Words ', sorted(count_mt5.items()))


count_mt5 = sorted(count_mt5.items())
count_mt5_l , _ = zip(*count_mt5)

print('Average mT5', np.average(preds_mt5_l))



f, ax = plt.subplots()

plt.plot(count_l[3:23], preds_l[3:23],'g', label='ByT5')
plt.plot(count_mt5_l[3:23], preds_mt5_l[3:23],'b', label='mT5')
plt.plot(count_l[3:23], ocrs_l[3:23],'r', label='OCR')

plt.xlabel("Word Length")
plt.ylabel("Character Error Rate")
# plt.yticks(np.arange(0, 11, 1))

ax.set_xticks([5,10,15,20])
plt.legend(loc='upper center', ncol=3) #bbox_to_anchor =(0.8, 0.7)

rect = plt.Rectangle((7.5,74.5), width=11.5, height=9.5,fill=True, facecolor="lightblue", clip_on=False)
ax.add_patch(rect)

ax.text(8, 80 , "Overall CER :" , fontsize=12.0)
ax.text(13, 80, "19.1 ", c='g', fontsize=12.0)
ax.text(15, 80, "25.3 ", c='b', fontsize=12.0)
ax.text(17, 80, "45.8 ", c='r', fontsize=12.0)

ax.text(8, 76 , "Overall WER :" , fontsize=12.0)
ax.text(13, 76, "21.3 ", c='g', fontsize=12.0)
ax.text(15, 76, "32.7 ", c='b', fontsize=12.0)
ax.text(17, 76, "50.4 ", c='r', fontsize=12.0)

plt.savefig('word_wise_cer.png')

