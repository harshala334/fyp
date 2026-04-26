# 🕵️‍♂️ DAM-CMA: Multimodal Fake News Detection
### Domain Adversarial Learning & Cross-Modal Attention Framework

This repository contains the official implementation of the **DAM-CMA** framework for identifying multimodal misinformation. The system combines textual analysis (BERT) with visual features (ResNet-50) using a sophisticated **Cross-Modal Attention (CMA)** mechanism and **Domain Adversarial Neural Networks (DANN)** for cross-source robustness.

---

## 🚀 Quick Start: Running the Demo
To launch the professional web interface for live news analysis:
```bash
# 1. Activate your environment
.\venv\Scripts\activate

# 2. Start the Streamlit app
streamlit run src/app/main.py
```

---

## 🏗️ Architecture Overview
- **Text Encoder**: BERT-base-uncased (Frozen)
- **Image Encoder**: ResNet-50 (Global Average Pooling)
- **Fusion Layer**: Cross-Modal Attention (CMA) - Automatically aligns keywords with image regions.
- **Robustness**: Domain Adversarial Training (DANN) - Ensures the model works across different news sources (PolitiFact, FactCheck, etc.).

## 📊 Dataset Statistics
The project utilizes a custom-scraped dataset of **2,250 articles**:
- **PolitiFact**: 1,800 articles
- **FactCheck.org**: 450 articles
- **Class Balance**: 1,200 Fake (53%) vs. 1,050 Real (47%) - **Balanced at the Data Level.**

## 🧪 Research Results (97.6% Accuracy)
| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **DAM-CMA (Proposed)** | **97.6%** | **0.97** |
| Simple Fusion | 95.3% | 0.94 |
| Image-Only | 95.0% | 0.94 |
| Text-Only | 68.5% | 0.61 |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/harshala334/fyp.git
cd fyp
```

### 2. Environment Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Training on Google Colab
For high-performance training using a GPU:
1. Open `Colab_Ready_Training.ipynb` in Google Colab.
2. Upload `training_data.zip`.
3. Run All Cells.
4. Download the trained `.pth` files and place them in the `models/` directory.

---

## 📁 Project Structure
- `src/app/main.py`: Streamlit Web Interface
- `src/models/dam_cma.py`: Main Research Model Architecture
- `src/utils/visualizer.py`: Attention Heatmap Generation
- `src/scrapers/`: Real-time data collection scripts
- `baselines/`: Comparative research models
- `RESEARCH_RESULTS.md`: Detailed performance analysis

---
**Author:** Harshala  
**Research Topic:** Domain Adversarial Learning & Cross-Modal Attention for Fake News Detection
