from torch import nn


class SentimentModel(nn.Module):
    def __init__(self, embedding, config):
        super(SentimentModel, self).__init__()

        self.bert = embedding
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # last_hidden_state = bert_out.last_hidden_state
        pooled_output = bert_out.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits