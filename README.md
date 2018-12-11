### Original Source:
We used a small part of the dataset downloading and preprocessing code released by the official body that released the VQA dataset.
link: https://github.com/GT-Vision-Lab/VQA_LSTM_CNN/blob/master/data/vqa_preprocessing.py
This file is used as the vqa_preprocess.py file in our codebase

### List of files modified
Every part of the codebase was developed by the team except for the already noted preprocessing files mentioned above.

### Commands to train and test the model

Given the current configuration, we can train the models using the following steps

1. baseline model: run 
```python main_baseline.py```
2. single word model: run 
```python main_one_word.py```
3. multi word model: run 
```python main_enc_dec.py```

For demo of our models:
```python demo.py```

All files take command line arguments that have some default value set, the parameters can be tweaked as the user likes

**Machine and software requirements**
- GPU access (compulsory for demo)
- Pytorch v0.4
- torchtext (by pytorch)
- Python 3.7+
- SPaCy
- CUDA toolkit 9.0
- CuDnn
