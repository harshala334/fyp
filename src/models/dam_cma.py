import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision import models
from torch.autograd import Function

# 1. The Gradient Reversal Layer (GRL)
# Reverses gradients during backward pass to confuse the domain classifier
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# 2. Cross-Modal Attention (CMA)
# Finds the relationship between specific words and image regions
class CMA_Fusion(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, shared_dim=512):
        super(CMA_Fusion, self).__init__()
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.img_proj = nn.Linear(img_dim, shared_dim)
        self.query = nn.Linear(shared_dim, shared_dim)
        self.key = nn.Linear(shared_dim, shared_dim)
        self.value = nn.Linear(shared_dim, shared_dim)

    def forward(self, text_feats, img_feats):
        T = F.relu(self.text_proj(text_feats))
        I = F.relu(self.img_proj(img_feats))

        # Text-Guided Image Attention
        Q = self.query(T)
        K = self.key(I)

        # Calculate scores: [Batch, Seq_Len, Num_Regions]
        scores = torch.bmm(Q, K.transpose(1, 2)) / (512**0.5)
        attn_weights = F.softmax(scores, dim=-1)

        V = self.value(I)
        fused_context = torch.bmm(attn_weights, V)

        # Combine Text + Attended Image
        combined = torch.cat([T, fused_context], dim=-1)
        final_vector = torch.mean(combined, dim=1) # Pooling

        return final_vector, attn_weights

# 3. The Main DAM-CMA Model
class DAM_CMA_Model(nn.Module):
    def __init__(self, num_domains=2, freeze_encoders=True):
        super(DAM_CMA_Model, self).__init__()

        # Encoders
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        if freeze_encoders:
            for param in self.bert.parameters(): param.requires_grad = False
            for param in self.resnet_backbone.parameters(): param.requires_grad = False

        # Fusion Layer
        self.cma = CMA_Fusion()

        # Task Classifier (Fake vs Real)
        self.label_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # Domain Classifier (Adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains)
        )

    def forward(self, input_ids, attention_mask, pixel_values, alpha=1.0):
        # 1. Feature Extraction
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_out.last_hidden_state

        img_out = self.resnet_backbone(pixel_values)
        img_feats = img_out.view(img_out.size(0), 2048, -1).permute(0, 2, 1)

        # 2. Cross-Modal Fusion
        fused_vec, attn_weights = self.cma(text_feats, img_feats)

        # 3. Classification
        label_pred = self.label_classifier(fused_vec)

        # 4. Domain Adversarial Training (GRL)
        reverse_vec = GradientReversalFn.apply(fused_vec, alpha)
        domain_pred = self.domain_classifier(reverse_vec)

        return label_pred, domain_pred, attn_weights
