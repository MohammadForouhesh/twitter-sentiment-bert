from sentence_transformers import SentenceTransformer
import torch

print('torch version:\t', torch.__version__)

MAX_LEN  = 280
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE  = 128
LABEL_LIST       = ['sad', 'meh', 'happy']
MODEL_NAME_OR_PATH = 'HooshvareLab/albert-fa-zwnj-base-v2'

id2label = {0: 'sad', 1: 'meh', 2: 'happy'}
label2id = {'sad': 0, 'meh': 1, 'happy': 2}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

file = open('persian.txt', 'r', encoding="utf8")
sw_persian = list(file.read().splitlines())

best_validation_loss = float('inf')
