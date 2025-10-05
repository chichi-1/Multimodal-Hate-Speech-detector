# Multimodal-Hate-Speech-detector
This project implements and evaluates a multimodal hate meme detection system capable of classifying memes in English and indigenous Nigerian languages as hate/non-hate.  It introduces a comparative framework between a baseline model (DistilBERT + ResNet) and a large-scale multimodal LLM (LLaVA), optimized using Particle Swarm Optimization (PSO). 
Multimodal Hate Meme Detection in Indigenous Nigerian Languages
A Thesis Project by CHINYERE ONUIGWE

MSc in Data Science — PAN-ATLANTIC UNIVERSITY

# Overview
This project implements and evaluates a multimodal hate meme detection system capable of classifying memes in English and indigenous Nigerian languages (Igbo, Yoruba, Hausa) as hate or non-hate.
It introduces a comparative framework between a lightweight baseline model (DistilBERT + ResNet) and a large-scale multimodal large language model (LLaVA), optimized using Particle Swarm Optimization (PSO).
The system aims to promote fairness-aware, culturally mindful, and computationally efficient AI for low-resource linguistic contexts.

# Research Aim
To develop a culturally mindful and efficient multimodal framework for detecting hate speech in memes across English and indigenous Nigerian languages by leveraging LLaVA with metaheuristic hyperparameter optimization, and comparing its performance against a lightweight DistilBERT + ResNet baseline.


# Project Structure
/content/drive/MyDrive/Thesis/
│
├── dataset/
│   ├── images/                  # Meme images (English + Indigenous)
│   ├── train.json               # Mapping of ID, text, image path, and label
│   ├── labels.json              # Hate vs. Non-hate labels
│   ├── ocr_results.csv          # OCR extracted text and quality metrics
│
├── llava_models/                # Pretrained and fine-tuned LLaVA models
│
├── output/
│   └── llava-hate-detector/     # Final trained model checkpoints and logs
│
├── Multimodal_Hate_Meme_Detection_.ipynb  # Full Colab notebook
├── Thesis.docx                  # Thesis documentation
└── README.md                    # Project description (this file)

# System Architecture

Pipeline Overview:

Meme Input → OCR (EasyOCR) → Text + Image Encoding
      ↓
  Baseline: DistilBERT + ResNet
      ↓
  Advanced: LLaVA (v1.5-7B)
      ↓
  Multimodal Fusion (Late / Transformer Fusion)
      ↓
  DNN Classifier → PSO Hyperparameter Optimization
      ↓
  Output: Hate / Non-hate Prediction + Fairness Metrics

# Model Components
Component	Description
OCR	Extracts embedded text from meme images using EasyOCR.
Text Encoder (Baseline)	DistilBERT — lightweight transformer for contextual embedding.
Image Encoder (Baseline)	ResNet — CNN-based visual feature extractor.
Multimodal Model	LLaVA-v1.5-7B — integrates CLIP-based vision encoder and LLaMA language model.
Fusion Strategies	Late Fusion (Baseline) / Transformer Fusion (LLaVA).
Classifier	Deep Neural Network (DNN) for final prediction.
Optimizer	Particle Swarm Optimization (PSO) for tuning: lr, flr, clr, wd, dr, hd, bs.

# PSO Best Parameters:
{'lr': 0.000752, 'flr': 0.2231, 'clr': 0.5862, 'wd': 8.19e-05, 'dr': 0.3472, 'hd': 417, 'bs': 11}

# Experimental Results
Model Comparison
Model	Accuracy	Precision	Recall	F1	AUC	F1 Std (Fairness)
Baseline (Late Fusion + DNN)	0.747	0.753	0.747	0.747	0.840	0.0598
LLaVA (Transformer Fusion + PSO-DNN)	0.761	0.764	0.761	0.761	0.862	0.0639

# Cross-Lingual F1 Scores
Language	Baseline F1	LLaVA (PSO) F1
English	0.846	0.808
Igbo	0.979	0.930
Yoruba	0.985	0.982
Hausa	0.988	0.929

# Key Findings
Baseline excels in indigenous languages; LLaVA shows balanced English–indigenous performance.
LLaVA achieves slightly higher global accuracy but no statistically significant improvement (p = 0.218).
Fairness (F1 Std ≈ 0.06) maintained across both models.
PSO improved stability and cross-lingual fairness.

# Key Insights
Bigger is not always better: The baseline model rivaled LLaVA without optimization, proving efficiency in low-resource contexts.
LLaVA’s strength is scalability: It simplifies joint text–image processing for large datasets.
OCR matters: Text extraction improved meme comprehension but diacritic loss (in Yoruba/Igbo) remains a challenge.
Fairness-aware design: Cross-lingual evaluation ensured equitable performance across all languages.

# Technologies Used
Python 3.11 (Google Colab environment)
Transformers (Hugging Face)
PyTorch
EasyOCR
LLaVA (LLaVA-v1.5-7B)
scikit-learn
Matplotlib / Seaborn
PSO Metaheuristic (PySwarms)

# Citation

If you use or build upon this work, please cite as:

Chinyere Onuigwe. (2025). Multimodal Hate Meme Detection in Indigenous Nigerian Languages Using Metaheuristics and LLaVA. MSc Thesis, Pan-Atlantic University.

# Contact

Author: Chinyere Onuigwe
Email: chinyere.onuigwe@gmail.com
Institution: Pan-Atlantic University, Department of Computer Science]
