from sentence_transformers import SentenceTransformer
import torch

N_EPOCH = 10
print(torch.__version__)
emb_model = SentenceTransformer('distilbert-base-multilingual-cased')#'paraphrase-multilingual-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

file = open('persian.txt', 'r', encoding="utf8")
sw_persian = list(file.read().splitlines())

best_validation_loss = float('inf')