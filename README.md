# 🌳 Bank Marketing Prediction using Decision Tree – Task 3 (PRODIGY_DS)

This repository contains **Task 3 of 5** for my Data Science Internship at **Prodigy InfoTech**.

---

## 🎯 Task Objective

**Task 3:** Build and visualize a decision tree classifier using a real-world dataset.

For this task, I worked with the **Bank Marketing dataset** to predict whether a client would subscribe to a term deposit (`y` column) using a decision tree classifier and evaluated the model's performance.

---

## 🧪 Dataset Used

- **File:** `bank-additional-full.csv`  
- **Source:** UCI Machine Learning Repository  
- **Description:** This dataset contains marketing campaign data of a Portuguese bank institution.

---

## 🛠️ Technologies & Libraries

- Python  
- Pandas  
- scikit-learn (LabelEncoder, DecisionTreeClassifier, metrics)  
- Matplotlib  

---

## 🔍 Workflow Overview

1. **Data Preprocessing**
   - Categorical features were encoded using `LabelEncoder`
   - Data was split into training (80%) and testing (20%) sets

2. **Model Training**
   - A `DecisionTreeClassifier` was trained on the dataset

3. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report

4. **Visualization**
   - The trained decision tree is visualized using `matplotlib.pyplot` and `sklearn.tree.plot_tree`

---

## 🚀 How to Run

1. Clone this repository  
2. Place `bank-additional-full.csv` in the root folder  
3. Run the following command:

```bash
python main.py
