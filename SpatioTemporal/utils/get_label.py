import torch 
from transformers import BertTokenizer, BertModel
import ast


def get_label_emb(labels_str, out_dim=128, device='cuda'):

    labels = ast.literal_eval(labels_str)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    projection = torch.nn.Linear(768, out_dim).to(device)

    labels_embedding = []
    for label in labels:
        inputs = tokenizer(label, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        cls_emb = projection(cls_emb)  # [1, 768]
        labels_embedding.append(cls_emb.squeeze(0))  # [768]
    
    labels_emb = torch.stack(labels_embedding, dim=0)  # [label_num, out_dim]
    return labels_emb  # [label_num, out_dim