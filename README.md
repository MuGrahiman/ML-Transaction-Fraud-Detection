
# **Transaction-Fraud-Detection** 🚀  

## **📖 Overview**  
This project implements a **Fraud Detection System** using **Random Forest Classifier** to classify transactions as **fraudulent or non-fraudulent** based on historical transaction data. The model analyzes transaction patterns and identifies anomalies that may indicate fraud.

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
- Execute all cells sequentially to train and evaluate the fraud detection model.  

---

## **🛠 Model Implementation**
The fraud detection model follows these key steps:  

✅ **Load & Preprocess Data**  
✅ **Split Data into Training & Testing Sets** (80% train, 20% test)  
✅ **Train a Random Forest Classifier**  
✅ **Make Predictions on Test Data**  
✅ **Evaluate Model Performance**  

### **📌 Model Used**: Random Forest Classifier  
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## **📈 Model Performance**
The trained model achieved **high accuracy in fraud detection**:  

### **✅ Accuracy Score**
```
Accuracy: 1.00
```

### **✅ Classification Report**
```
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      5541
         1.0       0.95      0.87      0.91        23

    accuracy                           1.00      5564
   macro avg       0.98      0.93      0.95      5564
weighted avg       1.00      1.00      1.00      5564
```

### **✅ Confusion Matrix**
```
[[5540    1]
 [   3   20]]
```

---

## **📌 Key Insights**
- **High Accuracy**: The model performs well in distinguishing fraud vs. non-fraud transactions.  
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
