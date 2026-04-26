import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class SimpleFusionBaseline(nn.Module):
    """
    Baseline Model: Simple Concatenation of BERT and ResNet features.
    NO Cross-Modal Attention.
    """
    def __init__(self, text_dim=768, img_dim=2048):
        super(SimpleFusionBaseline, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(self.resnet.children())[:-1]) # GAP output
        
        # Freeze encoders for fair comparison
        for param in self.bert.parameters(): param.requires_grad = False
        for param in self.resnet_backbone.parameters(): param.requires_grad = False
        
        # Classifier Head (Input = 768 + 2048 = 2816)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + img_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # 1. Text Features (CLS token)
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_out.pooler_output # [Batch, 768]
        
        # 2. Image Features (GAP)
        img_out = self.resnet_backbone(pixel_values)
        img_feats = img_out.view(img_out.size(0), -1) # [Batch, 2048]
        
        # 3. Concatenate
        combined = torch.cat([text_feats, img_feats], dim=-1)
        
        # 4. Classify
        logits = self.classifier(combined)
        return logits
