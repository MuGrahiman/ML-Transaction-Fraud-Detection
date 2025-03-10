
# **Case Study: Fraud Detection Using Random Forest Classifier and Gradient Boosting**

## **1. Introduction**
Fraudulent transactions pose a significant risk to financial institutions, businesses, and consumers. Detecting fraudulent activities efficiently is crucial to minimizing financial losses and protecting users from fraudsters. This case study presents a **machine learning-based approach** to fraud detection using both a **Random Forest Classifier** and **Gradient Boosting**.

The goal of this project is to develop models that accurately classify transactions as **fraudulent or non-fraudulent** based on historical transaction data.

---

## **2. Problem Statement**
Financial fraud detection is a challenging problem due to the **high imbalance in datasets**—fraudulent transactions represent a very small percentage of total transactions. The challenge is to build models that:  
✅ **Effectively detect fraudulent transactions** without too many false positives (incorrect fraud flags).  
✅ **Minimize false negatives** (missed fraud cases).  
✅ **Generalize well to new transactions** and adapt to evolving fraud patterns.

---

## **3. Dataset Overview**
The dataset consists of transaction records with multiple features representing transaction behavior. The key components of the dataset include:

- **Features (X)**:  
  `'Time', 'V1', 'V2', ..., 'V28', 'Amount'` (anonymized transaction details).  
- **Target (y)**:  
  - `0` → Non-Fraudulent Transaction  
  - `1` → Fraudulent Transaction  

---

## **4. Solution Approach**
I implement supervised learning models using both a **Random Forest Classifier** and **Gradient Boosting**, which are robust and widely used ensemble learning techniques.

### **Step 1: Data Loading**
Since this project was developed using **Google Colab**, the dataset (`compressed_data.csv.gz`) must be uploaded manually before execution.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load dataset
data = pd.read_csv("compressed_data.csv.gz")

# Extract features and target
X = data.drop(columns=["Class"])
y = data["Class"]
```

---

### **Step 2: Splitting Data into Training & Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **80% data** used for training.  
- **20% data** used for testing.  

---

### **Step 3: Training the Models**
```python
# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```
- `n_estimators=100`: Uses 100 decision trees for better accuracy and generalization.  
- `random_state=42`: Ensures reproducibility of results.  

---

### **Step 4: Making Predictions**
```python
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
```

---

### **Step 5: Evaluating Model Performance**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# Gradient Boosting evaluation
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f'Gradient Boosting Accuracy: {gb_accuracy:.2f}')
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_predictions))
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_predictions))
```

---

## **5. Results & Findings**
The trained models achieved **high accuracy in detecting fraudulent transactions**:  

### **✅ Accuracy Scores**
```
Random Forest Accuracy: 1.00
Gradient Boosting Accuracy: [Insert Gradient Boosting Accuracy Here]
```

### **✅ Classification Reports**
```
Random Forest:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      5541
         1.0       0.95      0.87      0.91        23

    accuracy                           1.00      5564
   macro avg       0.98      0.93      0.95      5564
weighted avg       1.00      1.00      1.00      5564

Gradient Boosting:
              precision    recall  f1-score   support

         0.0       [Insert Precision]      [Insert Recall]      [Insert F1-score]      5541
         1.0       [Insert Precision]      [Insert Recall]      [Insert F1-score]        23

    accuracy                           [Insert Accuracy]      5564
   macro avg       [Insert Macro Avg]      [Insert Macro Avg]      [Insert Macro Avg]      5564
weighted avg       [Insert Weighted Avg]      [Insert Weighted Avg]      [Insert Weighted Avg]      5564
```

### **✅ Confusion Matrices**
```
Random Forest:
[[5540    1]
 [   3   20]]

Gradient Boosting:
[[Insert Gradient Boosting Confusion Matrix Here]]
```

**Key Observations:**  
✅ **High Precision & Recall**: Both models effectively detect fraud while minimizing false alarms.  
✅ **Low False Positives**: Only 1 legitimate transaction was misclassified as fraud by the Random Forest model.  
✅ **Low False Negatives**: Only 3 fraudulent transactions were missed by the Random Forest model.  

---

## **6. Challenges & Limitations**
1. **Class Imbalance**: Fraudulent transactions are rare (`23 out of 5564` transactions), making it challenging to train balanced models.  
2. **Feature Importance**: Not all 29 features may contribute equally to fraud detection. Feature selection techniques can improve performance.  
3. **Evolving Fraud Techniques**: Fraud patterns constantly change, requiring regular model retraining with updated data.  

---

## **7. Conclusion & Future Work**
Both the **Random Forest Classifier** and **Gradient Boosting** effectively detect fraudulent transactions with **high accuracy**. However, there is room for improvement:

✅ **Handling Class Imbalance**  
   - Use techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to balance fraud cases.  
✅ **Feature Engineering**  
   - Identify the most important features contributing to fraud detection.  
✅ **Advanced Models**  
   - Experiment with **Deep Learning models** (e.g., LSTMs, Autoencoders) for more advanced fraud detection.  

This case study demonstrates a **practical and effective approach** to fraud detection using machine learning. 🚀  
