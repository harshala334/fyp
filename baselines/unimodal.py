import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class TextOnlyBaseline(nn.Module):
    """
    Baseline Model: Uses ONLY BERT for text analysis.
    Ignores image data completely.
    """
    def __init__(self, text_dim=768):
        super(TextOnlyBaseline, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze encoder for consistency
        for param in self.bert.parameters(): param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values=None):
        # pixel_values is ignored
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_out.pooler_output # [Batch, 768]
        return self.classifier(text_feats)

class ImageOnlyBaseline(nn.Module):
    """
    Baseline Model: Uses ONLY ResNet-50 for image analysis.
    Ignores text data completely.
    """
    def __init__(self, img_dim=2048):
        super(ImageOnlyBaseline, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(self.resnet.children())[:-1]) # GAP output
        
        # Freeze encoder for consistency
        for param in self.resnet_backbone.parameters(): param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # input_ids and mask are ignored
        img_out = self.resnet_backbone(pixel_values)
        img_feats = img_out.view(img_out.size(0), -1) # [Batch, 2048]
        return self.classifier(img_feats)
