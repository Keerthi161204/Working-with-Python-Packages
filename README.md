# ML Exercise 1 – Working with Python Packages

This repository contains **Experiment 1** from the *Machine Learning Algorithms Laboratory*, focused on exploring essential **Python libraries for Machine Learning**, performing **Exploratory Data Analysis (EDA)**, and applying **basic preprocessing techniques** on real-world datasets.

---

## 📌 Experiment Details

- **Title:** Working with Python Packages  
- **Course:** ICS1512 – Machine Learning Algorithms Laboratory  
- **Degree & Branch:** B.E. Computer Science & Engineering  
- **Semester:** V  
- **Academic Year:** 2025–2026 (Odd)  
- **Batch:** 2023–2028  

---

## 🎯 Aim

To explore popular Python libraries such as **NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn** using real-world datasets and perform:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature analysis
- Train–test data splitting

---

## 🧰 Libraries Used

- **NumPy** – Numerical computations  
- **Pandas** – Data manipulation and analysis  
- **Matplotlib** – Data visualization  
- **Seaborn** – Statistical data visualization  
- **Scikit-Learn** – Data preprocessing and ML utilities  

---

## 📂 Datasets Used

The following datasets are referenced in this experiment:

- Loan Amount Prediction  
- Iris Dataset  
- Predicting Diabetes  
- Email Spam Classification  
- Handwritten Character Recognition (MNIST)  

> **Primary dataset used in the code:**  
> **Loan Approval Dataset** (`loan_approval_dataset.csv`)

---

## 🤖 Type of Machine Learning Task

| Dataset | ML Task Type |
|------|--------------|
| Loan Amount Prediction | Supervised Regression |
| Iris Dataset | Classification |
| Predicting Diabetes | Classification |
| Email Spam Classification | Classification |
| Handwritten Recognition (MNIST) | Classification |

---

## 🧪 Experiment Workflow

### 1️⃣ Data Loading
- Load dataset using Pandas
- Inspect data using `.head()`, `.info()`, `.describe()`

### 2️⃣ Exploratory Data Analysis (EDA)
- Pair plots for feature relationships
- Histograms for feature distributions
- Bar charts for categorical variables
- Scatter plots against loan status
- Correlation heatmap
- Box plots for outlier detection

### 3️⃣ Data Preprocessing
- Missing value handling using median imputation
- Outlier treatment using quantile clipping
- Duplicate record removal
- Encoding categorical variables using `LabelEncoder`
- Feature scaling using `RobustScaler`

### 4️⃣ Train–Test Split
- Dataset split into training and testing sets using `train_test_split`

---

## 📊 Visualizations Generated

- Pair Plot of Features  
- Histogram Distribution of Numerical Features  
- Loan Status Bar Chart  
- Scatter Plots (Income vs Loan Status)  
- Feature Correlation Heatmap  
- Box Plot for Feature Distribution Comparison  

---

## 🧠 ML Task Identification Summary

| Dataset | ML Task | Feature Selection | Algorithm |
|------|-------|------------------|-----------|
| Iris Dataset | Classification | ANOVA | KNN, SVM |
| Loan Amount Prediction | Regression | SelectKBest | Linear Regression |
| Predicting Diabetes | Classification | Chi-Square | Logistic Regression |
| Email Spam Classification | Classification | Chi-Square | Naive Bayes |
| Handwritten Recognition (MNIST) | Classification | PCA | CNN |

---

## 📈 Inference

This experiment demonstrates how Python’s ML ecosystem supports:

- Efficient data loading and manipulation  
- Visual understanding of datasets  
- Data cleaning and preprocessing  
- Feature encoding and scaling  
- Preparing datasets for machine learning models  

---

## 🎓 Learning Outcomes

After completing this experiment, you will be able to:

- Perform data analysis using **Pandas**
- Create meaningful visualizations using **Matplotlib** and **Seaborn**
- Apply preprocessing techniques like encoding and scaling
- Identify suitable ML task types for datasets
- Prepare datasets for machine learning pipelines

---

## ▶️ How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/ML-Exercise-1.git
