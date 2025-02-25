# 🏥 Hepatitis C Disease Detection using SMOTE, Optuna, and SHAP 🚀

## 👨‍💻 Authors

**Nirmalkumaar T K, S. M. Mehzabeen, Dr. R. Gayathri, Pratosh Karthikeyan, Pranav A, Pranav Manikandan Sundaresan** 🏆  
📅 Date: June 2023 - October 2023  
📧 Contact: thirupallikrishnan.n@northeastern.edu 
🔗 [LinkedIn](https://www.linkedin.com/in/nirmalkumartk/) | [GitHub](https://github.com/NirmalKumar31)

## 🖥️ Tech Stack
- 🐍 Python
- 📊 Pandas, NumPy
- 📈 Matplotlib, Seaborn
- 🤖 Scikit-Learn, Optuna, SHAP
- ⚡ SMOTE (Synthetic Minority Over-sampling Technique)
- 📦 imbalanced-learn

## 📌 Overview
This repository contains the implementation of a **machine learning model** for **Hepatitis C disease detection**. The goal is to enhance **early detection** and **classification accuracy** by utilizing advanced techniques like **SMOTE, Optuna, and SHAP**.

## 🌟 Key Contributions
✅ **Enhanced Hepatitis C detection** by applying **SMOTE** for handling class imbalance.  
✅ **Optimized model performance** using **Optuna for hyperparameter tuning**.  
✅ **Improved interpretability** through **SHAP (SHapley Additive exPlanations)**.  
✅ **Conducted extensive Exploratory Data Analysis (EDA)** for feature selection and insights.  
✅ **Achieved an accuracy of 97% using an ensemble model** consisting of multiple classifiers.  

## 📊 Dataset
The dataset used for this research is sourced from the **UCI Machine Learning Repository**:
🔗 [HCV Dataset](https://archive.ics.uci.edu/ml/datasets/HCV+data)

### 📋 Data Description
The dataset contains **laboratory results and demographic data** from blood donors and Hepatitis C patients. It includes:
- **Healthy individuals**
- **Patients with Hepatitis C**
- **Individuals with fibrosis**
- **Cirrhosis patients**

### 🔢 Feature List
The dataset consists of **numerous biochemical and hematological markers**, including:
- **Age**
- **Albumin (ALB)**
- **Aspartate Aminotransferase (AST)**
- **Alanine Aminotransferase (ALT)**
- **Alkaline Phosphatase (ALP)**
- **Bilirubin (BIL)**
- **Cholinesterase (CHE)**
- **Cholesterol (CHOL)**
- **Creatinine (CREA)**
- **Gamma-Glutamyl Transferase (GGT)**
- **Protein (PROT)**

## 🛠️ Methodology
### 1️⃣ Exploratory Data Analysis (EDA)
📊 Visualized class distributions (showing severe class imbalance)  
📌 Identified key correlated features using a **heatmap**  
📈 Analyzed liver function indicators and their impact on Hepatitis C classification  

### 2️⃣ Data Preprocessing
🔄 **Handled missing values** and standardized numerical features.  
📊 **Balanced the dataset** using **SMOTE** to improve classification performance.  
🔬 **Selected relevant features** through correlation analysis and SHAP values.  

### 3️⃣ Model Selection
Several machine learning models were evaluated:
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes Classifier**
- **Multilayer Perceptron (MLP)**
- **Ensemble Model with Voting Classifier**

### 4️⃣ Hyperparameter Optimization using Optuna
To fine-tune our models and **boost accuracy**, we leveraged **Optuna** for automated hyperparameter tuning. It helped:
- Find the **optimal hyperparameter configurations**.
- Improve the **generalization ability** of models.
- Reduce **overfitting** and increase stability.

### 5️⃣ Model Performance
| 🏆 Model                        | 🎯 Accuracy (%) |
|---------------------------------|----------------|
| 📊 Logistic Regression          | 88.17%        |
| 🌲 Random Forest Classifier     | 94.62%        |
| 🔍 K-Nearest Neighbors (KNN)   | 92.47%        |
| 🧪 Naive Bayes                 | 94.63%        |
| 🤖 Support Vector Machines     | 94.62%        |
| 🏅 Multilayer Perceptron (MLP) | 95.69%        |
| ⭐ **Optimized Ensemble Model** | **97.00%**    |

### 6️⃣ Model Explainability with SHAP
📌 Used **SHAP values** to interpret model predictions and **identify key predictors**.
📊 **Visualized feature importance** using **bar graphs and force plots**.
🔬 Provided healthcare professionals with **insights into how the model makes predictions**.

## 🚀 Installation & Usage
### 🔧 Prerequisites
Make sure you have Python installed (>=3.8) and the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna shap imbalanced-learn
```
Ensure you have the following Python packages installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna shap imbalanced-learn
```

### ▶️ Running the Model
1️⃣ Clone this repository:
   ```bash
   git clone https://github.com/NirmalKumar31/Enhancing-Hepatitis-C-disease-Predication-using-ML.git
   cd Enhancing-Hepatitis-C-disease-Predication-using-ML
   ```
2️⃣ Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook "Hepatits C disease detection - Ensemble Model.ipynb"
   ```
   📌 [View Notebook](https://github.com/NirmalKumar31/Enhancing-Hepatitis-C-disease-Predication-using-ML/blob/main/Hepatits%20C%20disease%20detection%20-%20Ensemble%20Model.ipynb)

## 🎯 Results & Conclusion
- **SMOTE significantly improved model performance** by balancing the dataset.
- **Optuna-optimized ensemble model achieved the highest accuracy (97%)**.
- **SHAP provided critical insights into key features affecting Hepatitis C detection.**
- The results indicate **great potential for AI-driven Hepatitis C diagnostics**.

## 🔮 Future Work
🚑 **Integration with clinical decision support systems**.  
🧠 **Exploring deep learning models for enhanced accuracy**.  
📱 **Developing a mobile application for remote diagnostics**.  

🔗 **[Enhancing Hepatitis C Disease Detection Research Paper](https://github.com/NirmalKumar31/Enhancing-Hepatitis-C-disease-Predication-using-ML/blob/bd2e2ef02c1f5c819e28a96c498dc4e43198b68f/Research%20Paper_%20Enhancing%20Hepatitis%20C%20Disease%20Detection_%20A%20Study%20Using%20SMOTE%2C%20Optuna%2C%20and%20SHAP%20%20(1).pdf)**  

## 📚 References
📖 Ahmed M. Elshewey et al. (2023). "hyOPTGB Model for Hepatitis C Prediction."  
📖 Ali Mohd Ali et al. (2023). "Explainable Machine Learning Approach for Hepatitis C Diagnosis."  

---
🎯 This project aims to contribute to **early Hepatitis C detection** and **improved medical diagnostics** through **AI-driven solutions**. 🏥💡

