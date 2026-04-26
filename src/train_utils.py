import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def train_model(model, loader, device, epochs=5, lr=2e-5, is_adversarial=False):
    """
    Unified training loop for all models (Baselines and DAM-CMA).
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # We use BCEWithLogitsLoss for binary label classification
    criterion_label = nn.BCEWithLogitsLoss()
    # We use CrossEntropyLoss for domain classification (multi-class source detection)
    criterion_domain = nn.CrossEntropyLoss() if is_adversarial else None

    print(f"Starting training on {device} (Adversarial: {is_adversarial})")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Calculate alpha for GRL if adversarial
        alpha = 0
        if is_adversarial:
            p = float(epoch) / epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1).float() # Ensure float for BCE
            
            optimizer.zero_grad()

            # Forward Pass
            if is_adversarial:
                label_pred, domain_pred, _ = model(input_ids, mask, pixel_values, alpha=alpha)
                domain_labels = batch['domain'].to(device).long() # Ensure long for CrossEntropy
                
                loss_label = criterion_label(label_pred, labels)
                loss_domain = criterion_domain(domain_pred, domain_labels)
                loss = loss_label + loss_domain
            else:
                # Handle baselines which might not take all inputs
                try:
                    label_pred = model(input_ids, mask, pixel_values)
                except TypeError:
                    # Some unimodal models only take specific inputs
                    if hasattr(model, 'bert'):
                        label_pred = model(input_ids, mask)
                    else:
                        label_pred = model(pixel_values=pixel_values)
                
                loss = criterion_label(label_pred, labels)

            # Backward Pass
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            preds = torch.sigmoid(label_pred) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Acc: {avg_acc:.2f}%")

    print("Training Complete.")
    return model
