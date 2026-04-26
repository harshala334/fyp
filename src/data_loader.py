import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
import os

class MultimodalDataset(Dataset):
    def __init__(self, jsonl_path, max_len=128):
        """
        Custom Dataset for Fake News Detection.
        Loads unified master_data.jsonl.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found at {jsonl_path}")

        print(f"Loading dataset: {jsonl_path}")
        self.data = pd.read_json(jsonl_path, lines=True)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        
        # Standard ResNet Image Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Text Processing
        inputs = self.tokenizer(
            str(row['text']),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Image Processing
        img_path = row['image_path']
        # Note: image_path is relative to project root in our unified schema
        if isinstance(img_path, str) and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.img_transform(image)
            except:
                # Fallback to zero tensor if image corrupted
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'pixel_values': image,
            'label': torch.tensor(row['label'], dtype=torch.float),
            'domain': torch.tensor(row['domain'], dtype=torch.float)
        }

def get_dataloader(jsonl_path, batch_size=16, shuffle=True):
    dataset = MultimodalDataset(jsonl_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
