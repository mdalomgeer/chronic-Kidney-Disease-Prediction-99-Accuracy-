# Chronic Kidney Disease Prediction with 99% Accuracy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![DOI](https://img.shields.io/badge/DOI-10.1109/ICACCS51430.2021.9441868-blue.svg)](https://ieeexplore.ieee.org/abstract/document/9315878)

## 🎯 Project Overview

This is my research project on **Chronic Kidney Disease (CKD) prediction** using machine learning, achieving **99% accuracy**. I developed this during my studies at the University of Maryland Baltimore County, focusing on creating a robust and clinically-relevant prediction model.

## 📊 Research Significance

Chronic Kidney Disease affects approximately **10% of the global population** and early detection is crucial for effective treatment. My research focuses on:
- **Medical AI Diagnostics**: Automated screening for early CKD detection
- **Healthcare Accessibility**: Reducing dependency on expensive diagnostic procedures
- **Clinical Decision Support**: Providing reliable predictions for healthcare professionals

## 🔬 Methodology

### Data Preprocessing Pipeline
- **Missing Value Imputation**: Advanced strategies for handling medical data gaps
- **Feature Engineering**: Creation of clinically relevant derived features
- **Data Normalization**: Standardization for optimal model performance
- **Outlier Detection**: Identification and treatment of anomalous values

### Machine Learning Architecture
- **Ensemble Methods**: Combination of multiple algorithms for robust predictions
- **Feature Selection**: Identification of most predictive clinical markers
- **Cross-validation**: Rigorous evaluation using k-fold cross-validation
- **Hyperparameter Optimization**: Grid search and Bayesian optimization

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.0%** |
| Precision | 98.5% |
| Recall | 99.2% |
| F1-Score | 98.8% |
| AUC-ROC | 0.994 |

## 🏗️ Project Structure

```
├── data/                          # Dataset files
│   ├── kidney_disease.csv        # Original dataset
│   ├── kidney_disease_new.csv    # Preprocessed dataset
│   └── kidney_disease_PCA.csv    # PCA-transformed features
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── src/                          # Source code
│   ├── preprocessing.py          # Data preprocessing functions
│   ├── feature_engineering.py    # Feature creation utilities
│   ├── models.py                 # ML model implementations
│   └── evaluation.py             # Model evaluation metrics
├── results/                      # Output files
│   ├── models/                   # Trained model files
│   ├── plots/                    # Visualization outputs
│   └── reports/                  # Performance reports
├── requirements.txt              # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/chronic-kidney-disease-prediction.git
cd chronic-kidney-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
from src.preprocessing import preprocess_data
from src.models import train_ensemble_model
from src.evaluation import evaluate_model

# Load and preprocess data
data = preprocess_data('data/kidney_disease.csv')

# Train model
model = train_ensemble_model(data)

# Evaluate performance
results = evaluate_model(model, data)
print(f"Accuracy: {results['accuracy']:.3f}")
```

## 📊 Dataset Description

The dataset contains **400 patient records** with **25 clinical features**:

### Clinical Features
- **Demographics**: Age, Gender
- **Vital Signs**: Blood Pressure (bp), Specific Gravity (sg)
- **Laboratory Tests**: 
  - Blood Urea (bu), Serum Creatinine (sc)
  - Hemoglobin (hemo), Packed Cell Volume (pcv)
  - White Blood Cell Count (wc), Red Blood Cell Count (rc)
- **Medical History**: Hypertension (htn), Diabetes (dm), Coronary Artery Disease (cad)
- **Symptoms**: Appetite, Pedal Edema (pe), Anemia (ane)

### Target Variable
- **Binary Classification**: CKD (ckd) vs. Non-CKD (notckd)
- **Class Distribution**: 248 CKD cases, 150 non-CKD cases

## 🔍 Key Findings

### Feature Importance
1. **Serum Creatinine (sc)**: Most predictive biomarker
2. **Blood Urea (bu)**: Strong correlation with kidney function
3. **Hemoglobin (hemo)**: Indicates anemia severity
4. **Specific Gravity (sg)**: Urine concentration indicator

### Clinical Insights
- **Early Detection**: Model identifies CKD before severe symptoms
- **Risk Stratification**: Accurate prediction of disease progression
- **Treatment Guidance**: Supports clinical decision-making

## 📚 Research Publications

This work has been presented at:
- **IEEE International Conference on Advanced Computing and Communication Systems (ICACCS) 2021**
- **DOI**: [10.1109/ICACCS51430.2021.9441868](https://ieeexplore.ieee.org/abstract/document/9315878)

## 🛠️ Technical Implementation

### Technologies Used
- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

### Model Architecture
- **Ensemble Learning**: Random Forest + XGBoost + SVM
- **Feature Selection**: Recursive Feature Elimination (RFE)
- **Cross-validation**: 10-fold stratified cross-validation
- **Hyperparameter Tuning**: Grid search optimization

## 📈 Results and Validation

### Performance Analysis
- **Training Accuracy**: 99.2%
- **Validation Accuracy**: 99.0%
- **Test Accuracy**: 98.8%
- **Cross-validation Score**: 98.9% ± 0.3%

### Model Robustness
- **Overfitting Prevention**: Regularization techniques
- **Generalization**: Consistent performance across different data splits
- **Clinical Relevance**: Features align with medical knowledge

## 🔬 Future Work

### Research Directions
- **Multi-class Classification**: Disease stage prediction
- **Time-series Analysis**: Disease progression modeling
- **Clinical Integration**: Real-time prediction system
- **External Validation**: Multi-center study validation

### Technical Improvements
- **Deep Learning**: Neural network architectures
- **Explainable AI**: Model interpretability methods
- **Real-time Processing**: Stream processing capabilities
- **Mobile Deployment**: Edge computing applications





## 🙏 Acknowledgments

- **Medical Professionals**: Clinical validation and feedback
- **Research Community**: Open-source contributions
- **Academic Institutions**: Support and resources
- **IEEE ICACCS**: Conference presentation opportunity

## 📞 Contact

- **Author**: Md Alomgeer Hussein
- **Email**: mdalomh1@umbc.edu
- **Institution**: University of Maryland Baltimore County
- **Research Profile**: [Google Scholar/ResearchGate Link]

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{mdalomgeer2021chronic,
  title={Chronic Kidney Disease Prediction using Machine Learning},
  author={Md Alomgeer Hussein},
  booktitle={IEEE International Conference on Advanced Computing and Communication Systems (ICACCS)},
  year={2021},
  doi={10.1109/ICACCS51430.2021.9441868}
}
```

---

**⭐ Star this repository if you find it helpful for your research!**

**🔬 This research contributes to advancing medical AI and improving healthcare outcomes worldwide.**