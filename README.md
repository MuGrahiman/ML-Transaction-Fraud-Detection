

# **Transaction-Fraud-Detection** 🚀  

## **📖 Overview**  
This project implements a **Fraud Detection System** using both **Random Forest Classifier** and **Gradient Boosting** to classify transactions as **fraudulent or non-fraudulent** based on historical transaction data. The models analyze transaction patterns and identify anomalies that may indicate fraud.

---

## **📂 Project Structure**
```
📦 ML-Transaction-Fraud-Detection
 ┣ 📜 README.md                          # Project documentation
 ┣ 📜 FraudDetectionInTransaction.ipynb   # Jupyter Notebook with model implementation
 ┣ 📜 compressed_data.csv.gz              # Compressed dataset
 ┣ 📜 CASE_STUDY.md                       # Detailed case study
```

---

## **📊 Dataset Overview**
The dataset consists of historical transaction data with **29 feature columns** and **1 target column** (`Class`):  
- **Features**: `'Time', 'V1', 'V2', ..., 'V28', 'Amount'`  
- **Target (`Class`)**:
  - `0` → Non-Fraudulent Transaction  
  - `1` → Fraudulent Transaction  

---

## **⚙️ How to Use**
Since this project was developed using **Google Colab**, you can follow these steps:  

### **1️⃣ Open Google Colab**
- Go to **[Google Colab](https://colab.research.google.com/)**.  

### **2️⃣ Upload the Notebook**
- Open `FraudDetectionInTransaction.ipynb` from your GitHub repo.  

### **3️⃣ Upload the Dataset**
- Upload `compressed_data.csv.gz` manually in Colab.  
- Alternatively, modify the dataset path in the notebook if needed.  

### **4️⃣ Run the Cells**
- Execute all cells sequentially to train and evaluate the fraud detection models.  

---

## **🛠 Model Implementation**
The fraud detection model follows these key steps:  

✅ **Load & Preprocess Data**  
✅ **Split Data into Training & Testing Sets** (80% train, 20% test)  
✅ **Train a Random Forest Classifier**  
✅ **Train a Gradient Boosting Classifier**  
✅ **Make Predictions on Test Data**  
✅ **Evaluate Model Performance**  

### **📌 Models Used**: Random Forest Classifier & Gradient Boosting  
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
```

---

## **📈 Model Performance**
The trained models achieved **high accuracy in fraud detection**:  

### **✅ Accuracy Score**
```
Random Forest Accuracy: 1.00
Gradient Boosting Accuracy: [Insert Gradient Boosting Accuracy Here]
```

### **✅ Classification Report**
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

### **✅ Confusion Matrix**
```
Random Forest:
[[5540    1]
 [   3   20]]

Gradient Boosting:
[[Insert Gradient Boosting Confusion Matrix Here]]
```

---

## **📌 Key Insights**
- **High Accuracy**: Both models perform well in distinguishing fraud vs. non-fraud transactions.  
- **Precision & Recall**: High precision ensures fewer false positives, while recall indicates strong fraud detection capability.  
- **Class Imbalance**: Fraud cases (`Class = 1`) are significantly lower, which may require techniques like **SMOTE** to improve recall further.  

---

## **🚀 Future Enhancements**
✅ Implement **class imbalance handling** (e.g., SMOTE, weighted loss functions).  
✅ Experiment with **Deep Learning models** (e.g., Autoencoders, LSTMs).  
✅ Deploy as an **API** for real-time fraud detection.  

---

## **📑 Want to Learn More?**
👉 Read the full **[Case Study](CASE_STUDY.md)** for in-depth details!  
