import numpy as np
import torch


class TweetsDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Tweets. """

    def __init__(self, tokenizer, tweets, targets=None, label_list=None, max_len=128, device='cpu'):
        self.tokenizer = tokenizer
        self.tweets = tweets
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.max_len = max_len
        self.device = device
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])

        if self.has_target: target = self.label_map.get(str(self.targets[item]), str(self.targets[item]))

        encoding = self.tokenizer(tweet, add_special_tokens=True, truncation=True,
                                  max_length=self.max_len, return_token_type_ids=True, padding='max_length',
                                  return_attention_mask=True, return_tensors='pt')
        inputs = {'tweet': tweet,
                  'input_ids': encoding['input_ids'].flatten().to(self.device),
                  'attention_mask': encoding['attention_mask'].flatten().to(self.device),
                  'token_type_ids': encoding['token_type_ids'].flatten().to(self.device)}

        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long).to(self.device)

        return inputs


def create_data_loader(tokenizer, tweets, targets, max_len, label_list, batch_size, device):
    dataset = TweetsDataset(tokenizer=tokenizer, tweets=tweets, targets=targets, max_len=max_len, label_list=label_list,
                            device=device)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)