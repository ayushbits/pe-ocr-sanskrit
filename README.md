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

### Citation:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.07980,
  doi = {10.48550/ARXIV.2211.07980},
  url = {https://arxiv.org/abs/2211.07980},
  author = {Maheshwari, Ayush and Singh, Nikhil and Krishna, Amrith and Ramakrishnan, Ganesh},  
  title = {A Benchmark and Dataset for Post-OCR text correction in Sanskrit},
  publisher = {arXiv},
  year = {2022}
}
```
