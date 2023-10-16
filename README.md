## A Benchmark and Dataset for Post-OCR text correction in Sanskrit

> [A Benchmark and Dataset for Post-OCR text correction in Sanskrit](http://arxiv.org/abs/2211.07980)  
> Ayush Maheshwari, Nikhil Singh, Amrith Krishna and Ganesh Ramakrishnan                    
> Findings of EMNLP 2022

## Post-edited data
- *_devanagari.csv refers to train, test and validation split of manually post-edited OCR data
- ood-test.csv refers to out-of-domain test set consisting of 500 sentences as described in Section 4.1 of the paper.
### dev-transliterate-scripts/ 
- contains scripts to transliterate words from from SLP1 to Dev and vice-versa
## OCR images and their annotation
- OCR-Images-Annotation/ folder contains books containing test set of 500 images and their corresponding groundtruth. 
- BHS refers to Brahmastura Bhashyam
- GG refers to Grahalaghava of Ganesh Daivajna
- GOS refers to Goladhyaya

### Training Scripts
- Training scripts are present in the train-scripts directory


### Calculate CER, WER

- preds/ folder contains predictions and GT for the 500 sentences in out-of-domain test set
-  To calculate, run `pip install fastwer`
- python word_count_cer.py <cer/wer>

### Citation:
```bibtex
@inproceedings{maheshwari2022benchmark,
  title={A Benchmark and Dataset for Post-OCR text correction in Sanskrit},
  author={Maheshwari, Ayush and Singh, Nikhil and Krishna, Amrith and Ramakrishnan, Ganesh},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
  pages={6258--6265},
  year={2022}
}
```
