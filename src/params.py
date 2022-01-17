from sentence_transformers import SentenceTransformer
import torch

N_EPOCH = 10
print(torch.__version__)
emb_model = SentenceTransformer('distilbert-base-multilingual-cased') #'paraphrase-multilingual-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

file = open('persian.txt', 'r', encoding="utf8")
sw_persian = list(file.read().splitlines())

best_validation_loss = float('inf')

"""
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
"""