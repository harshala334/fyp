import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def generate_attention_heatmaps(model, batch, tokenizer, device='cpu', num_words=3):
    """
    Generates and plots explainability heatmaps for a trained DAM-CMA model.
    """
    model.eval()
    
    # 1. Prepare Batch
    input_ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    pixel_values = batch['pixel_values'].to(device)

    # 2. Find a valid sample in the batch (non-zero image)
    sample_idx = 0
    for i in range(pixel_values.shape[0]):
        if pixel_values[i].sum() != 0:
            sample_idx = i
            break

    # 3. Get Attention Weights
    with torch.no_grad():
        # Alpha=0 as we aren't training the domain classifier during inference
        _, _, attns = model(input_ids, mask, pixel_values, alpha=0)

    attn_map = attns[sample_idx] # [Seq_Len, 49]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[sample_idx])

    # 4. Filter for meaningful words
    valid_indices = [
        i for i, t in enumerate(tokens)
        if t not in ['[CLS]', '[SEP]', '[PAD]'] and not t.startswith('##')
    ]
    target_indices = valid_indices[:num_words]

    # 5. Denormalize Image for plotting
    img_tensor = pixel_values[sample_idx].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    # 6. Plotting
    plt.figure(figsize=(16, 5))
    
    # Plot Original
    plt.subplot(1, len(target_indices) + 1, 1)
    plt.imshow(img_display)
    plt.title("Original Image", fontweight="bold")
    plt.axis('off')

    # Plot Heatmaps
    for i, word_idx in enumerate(target_indices):
        word = tokens[word_idx]
        word_attn = attn_map[word_idx].cpu().numpy()
        
        # Reshape to 7x7 and upscale
        heatmap = word_attn.reshape(7, 7)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Normalize heatmap colors
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        plt.subplot(1, len(target_indices) + 1, i + 2)
        plt.imshow(img_display)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"Focus: '{word}'", fontweight="bold")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print Clean Claim
    clean_sentence = ' '.join([t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']]).replace(' ##', '')
    print(f"\nFull Analyzed Claim: {clean_sentence}")
