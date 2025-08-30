import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class CrossEncoderClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=4, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 4 policy labels


    def forward(self, input_ids, attention_mask, labels=None):
        # HuggingFace base models return: (last_hidden_state, pooled_output, ...)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch, num_labels]

        loss = None
        if labels is not None:
            # BCEWithLogits = better for multi-label (no softmax)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
    
    # save method
    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), f"{save_directory}/classifier.pt")

    # load method
    @classmethod
    def from_pretrained(cls, load_directory, model_name="microsoft/deberta-v3-base", num_labels=4):
        model = cls(model_name=model_name, num_labels=num_labels)
        model.encoder = AutoModel.from_pretrained(load_directory)
        model.classifier.load_state_dict(torch.load(f"{load_directory}/classifier.pt"))
        return model