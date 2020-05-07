

try:
    import tokenizers
    import os
    import torch
    print('Required Libraries for config.py Loaded')
except:
    print('Please make sure you have all the necessary Libraries for config.py !!')
    # pip install transformers tokenizers torch
    import tokenizers
    import os
    import torch
try:
    MODEL_TYPE = 'roberta'
    DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    N_SPLITS = 5
    PATH = '../input/roberta-base/'
    MODEL_PATH = f'{PATH}roberta-base-pytorch_model.bin'
    TRAINING_FILE = '../input/train.csv'
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
          vocab_file=f'{PATH}roberta-base-vocab.json',
          merges_file=f'{PATH}roberta-base-merges.txt',
          add_prefix_space=True,
          lowercase=True
      )
      NROWS = 10 if DEVICE=='cpu' else None
    print('Parameters Set in config.py.')
except:
    print('Error While setting parameters in config.py, Please check !!')

