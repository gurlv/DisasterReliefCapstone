# Disaster Response Message Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## AAI-590 Capstone Project | University of San Diego
**M.S. Applied Artificial Intelligence**

### Team Members
- Gurleen Virk (gvirk@sandiego.edu)
- Victor Hsu (vhsu@sandiego.edu)

---

## Project Overview

This project develops a **multi-label machine learning system** for automated classification of disaster response messages. During major disasters, emergency response organizations receive thousands of messages per hour, each requiring manual categorization across 36 distinct need categories. Our system reduces triage time from 3-5 minutes per message to under 1 second, enabling faster humanitarian response.

### Problem Statement
- **Challenge**: Manual message triage creates critical bottlenecks during disaster response
- **Scale**: 1,000+ messages per hour during major disasters
- **Complexity**: 36 category labels with severe class imbalance (some categories <2% prevalence)
- **Stakes**: Delayed categorization of medical/rescue needs can cost lives

### Solution
An optimized XGBoost-based multi-label classifier with:
- Per-label cost-sensitive learning for class imbalance
- Threshold tuning optimization (+9.1% F1 improvement)
- Hybrid deployment model for human-AI collaboration

---

## Key Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Micro F1 | 0.625 | 0.682 | +9.1% |
| Macro F1 | 0.461 | 0.499 | +8.2% |
| Precision | 0.582 | 0.657 | +12.9% |
| Recall | 0.674 | 0.708 | +5.0% |

### Critical Category Performance
| Category | F1 Score | Recall Improvement |
|----------|----------|-------------------|
| Food | 0.782 | - |
| Water | 0.711 | +20% |
| Shelter | 0.675 | +35% |
| Medical Help | 0.513 | +90% |
| Search & Rescue | 0.365 | +63% |

---

## Repository Structure

```
disaster-response-classification/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── notebooks/
│   ├── 01_EDA_Data_Cleaning.ipynb           # Data cleaning & exploratory analysis
│   ├── 02_Model_Training.ipynb              # Model training (XGBoost, Classifier Chains, DNN)
│   └── 03_Model_Optimization.ipynb          # Threshold tuning & hyperparameter optimization
│
├── data/
│   └── README.md                      # Dataset download instructions
│
├── models/
  └── README.md                      # Saved model files (not tracked in git)

```

---

## Notebooks Description

### 1. `01_EDA_Data_Cleaning.ipynb` 
**Purpose**: Data cleaning and exploratory data analysis

**Contents**:
- Data loading from Figure Eight disaster response dataset
- Missing value analysis and handling
- Text preprocessing pipeline (tokenization, lemmatization, stopword removal)
- Label distribution analysis and class imbalance visualization
- TF-IDF feature engineering (5,000 features)
- Label correlation analysis
- Train/validation/test split verification

**Key Outputs**:
- Cleaned datasets
- Preprocessor pipeline (saved as pickle)
- EDA visualizations

### 2. `02_Model_Training.ipynb`
**Purpose**: Model design, building, training, and comparison

**Contents**:
- **Model 1: XGBoost (Binary Relevance)** ✓ Selected
  - One classifier per label
  - Cost-sensitive learning with dynamic `scale_pos_weight`
  - Hyperparameters: 200 estimators, max_depth=5, learning_rate=0.2
  
- **Model 2: Classifier Chains**
  - Sequential label prediction capturing dependencies
  - Logistic Regression base classifier
  - Balanced class weights
  
- **Model 3: Deep Neural Network** ✓ Built from scratch using Keras/TensorFlow
  - Architecture: 512 → 256 → 128 → 36 (sigmoid)
  - Batch normalization and dropout (0.2-0.3)
  - Binary cross-entropy loss
  - Adam optimizer
  - Early stopping with patience=5

**Key Outputs**:
- Trained model files
- Training/validation curves
- Per-label performance metrics
- Model comparison visualizations

### 3. `03_Model_Optimization.ipynb` and `04_Further_Model_Optimization_and_Analysis`
**Purpose**: Advanced optimization and model analysis

**Contents**:
- **Threshold Tuning**
  - Per-label optimal threshold search (0.1-0.9 range)
  - F1-optimized thresholds
  - Recall-optimized thresholds for critical categories
  
- **Category-Specific Hyperparameter Search**
  - Grid search for critical categories (medical_help, water, food, shelter, search_and_rescue)
  - 108 hyperparameter combinations tested
  - 3.2 hours compute time
  
- **Final Model Analysis**
  - Error analysis by category
  - Confusion matrix analysis
  - Systematic error pattern identification

**Key Outputs**:
- Optimal threshold configuration (JSON)
- Final optimized model
- Performance comparison tables

---

## Dataset

**Source**: [Figure Eight Multilingual Disaster Response Messages](https://www.kaggle.com/datasets/landlord/multilingual-disaster-response-messages)

**Description**: Real disaster messages from major events including:
- Haiti Earthquake (2010)
- Chile Earthquake (2010)
- Pakistan Floods (2010)
- Hurricane Sandy (2012)

**Statistics**:
| Split | Messages |
|-------|----------|
| Training | 21,046 |
| Validation | 2,573 |
| Test | 2,629 |
| **Total** | **26,248** |

**Labels**: 36 binary category labels organized into:
- Critical Resources: food, water, shelter, clothing
- Emergency Services: medical_help, search_and_rescue, security
- Infrastructure: buildings, electricity, transport, hospitals
- Disaster Types: earthquake, floods, storm, fire, weather_related
- Social/Demographic: refugees, missing_people, death

**Challenges**:
- Severe class imbalance (search_and_rescue: 1.8%, water: 4.5%)
- 24.2% unlabeled messages
- 3 zero-support categories excluded (PII, offer, child_alone)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster neural network training)

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/disaster-response-classification.git
cd disaster-response-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.7.0
tensorflow>=2.10.0
keras>=2.10.0
nltk>=3.7
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
joblib>=1.1.0
```

---

## Usage

### Running the Notebooks

1. **Download the dataset** from Kaggle and place in `data/` directory

2. **Run notebooks in order**:
   ```bash
   jupyter notebook notebooks/01_EDA_Data_Cleaning.ipynb
   jupyter notebook notebooks/02_Model_Training.ipynb
   jupyter notebook notebooks/03_Model_Optimization.ipynb
   ```

### Using Pre-trained Models

```python
import joblib
import json

# Load model and preprocessor
models = joblib.load('models/optimized_xgboost_multilabel.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Load optimal thresholds
with open('models/optimal_thresholds.json', 'r') as f:
    thresholds = json.load(f)

# Predict on new message
message = "We need water and medical supplies urgently"
X = preprocessor.transform([message])

predictions = {}
for label, model in models.items():
    proba = model.predict_proba(X)[0, 1]
    threshold = thresholds[label]['threshold']
    predictions[label] = proba > threshold

print(predictions)
```

---

## Model Architecture

### Selected Model: XGBoost (Binary Relevance)

```
Input: TF-IDF Features (5,003 dimensions)
       ↓
┌─────────────────────────────────────────┐
│  34 Independent XGBoost Classifiers     │
│  (one per label)                        │
│                                         │
│  Hyperparameters:                       │
│  - n_estimators: 200                    │
│  - max_depth: 5                         │
│  - learning_rate: 0.2                   │
│  - scale_pos_weight: dynamic (neg/pos)  │
│  - subsample: 0.8                       │
│  - colsample_bytree: 0.8                │
└─────────────────────────────────────────┘
       ↓
Per-Label Threshold Application
       ↓
Output: 34 Binary Predictions
```

### Neural Network Architecture (Built from Scratch)

```
Input Layer (5,003 features)
       ↓
Dense(512) + BatchNorm + Dropout(0.3) + ReLU
       ↓
Dense(256) + BatchNorm + Dropout(0.3) + ReLU
       ↓
Dense(128) + BatchNorm + Dropout(0.2) + ReLU
       ↓
Dense(37) + Sigmoid
       ↓
Output: 37 Probability Scores
```

---

## Future Work

1. **BERT Embeddings**: Replace TF-IDF with fine-tuned DistilBERT for contextual understanding (expected +8-15% F1)

2. **Data Augmentation**: Address data scarcity through SMOTE, back-translation, and few-shot learning

3. **Production Pipeline**: Real-time API, active learning loop, drift monitoring, multilingual support

---

## References

1. Imran, M., Castillo, C., Diaz, F., & Vieweg, S. (2015). Processing social media messages in mass emergency: A survey. ACM Computing Surveys.

2. Figure Eight. (2019). Multilingual Disaster Response Messages. Kaggle.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.

4. Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine Learning.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- University of San Diego, Shiley-Marcos School of Engineering
- Figure Eight for providing the disaster response dataset
- Professor Anna Marbut for guidance
