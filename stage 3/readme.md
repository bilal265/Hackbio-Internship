# Depression Detection in University Students
A machine learning classification model to predict depression in university students based on various psychological, academic, and lifestyle factors.

## Table of Contents
Introduction
Dataset
Feature Selection & Engineering
Data Preprocessing
Model Selection & Training
Evaluation Metrics
Results
Ethical Considerations
Future Improvements
Installation & Usage
Contributors

## Introduction
Depression among university students is a growing concern, affecting their mental well-being and academic performance. This project aims to build a classification model that detects students with depression based on various features, including academic pressure, sleep duration, financial stress, and family history of mental illness.

The model can potentially be used in early intervention strategies to provide mental health support.

## Dataset
The dataset contains structured data with both numerical and categorical features. It includes:

Demographics: Gender, Age, Degree, Profession
Academic & Work Factors: Academic Pressure, Work Pressure, CGPA, Study/Work Hours
Lifestyle Factors: Sleep Duration, Dietary Habits
Mental Health Indicators: Suicidal Thoughts, Family History of Mental Illness
Depression Label: Target variable (Depressed / Not Depressed)

🔍 Handling Missing Values: Missing data was imputed using the most frequent values in each column.
📊 Class Imbalance Check: Depression cases were checked for imbalance.

## Feature Selection & Engineering
🚀 Important Features (From Random Forest Feature Importance):

Academic Pressure & Work Pressure
Financial Stress
Sleep Duration
Suicidal Thoughts
Family History of Mental Illness

### 📌 Potential Future Feature Engineering:
Combining Work Pressure + Academic Pressure → Total Stress Score
Creating a Mental Health Risk Score based on multiple features

## Data Preprocessing
Encoding Categorical Variables → Used LabelEncoder() for categorical features.
Scaling Numerical Features → Used StandardScaler() for normalization.
Handling Missing Values → Used SimpleImputer(strategy='most_frequent').
Splitting Data → train_test_split() with an 80-20 split.

## Model Selection & Training
We implemented and compared three classification models:

Logistic Regression (Baseline Model)
Random Forest Classifier (Feature Importance)
XGBoost Classifier (Advanced Boosting Model)

### 🔧 Hyperparameter Tuning for Random Forest:
Used GridSearchCV to optimize:
n_estimators: [50, 100, 200]
max_depth: [None, 10, 20]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]

## Evaluation Metrics
Given the sensitivity of depression detection, we prioritize:

✅ Recall (to reduce false negatives – missing depressed students)
✅ AUC-ROC Score (to measure model discrimination ability)
✅ F1-Score (to balance precision & recall)

## **Evaluation Metrics**  
Given the **sensitivity of depression detection**, we prioritize:  

✅ **Recall** (to reduce false negatives – missing depressed students)  
✅ **AUC-ROC Score** (to measure model discrimination ability)  
✅ **F1-Score** (to balance precision & recall)  

### **Model Performance Comparison**  

| Model                 | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|------------|--------|----------|---------|
| **Logistic Regression** | 0.86 | 0.88 | 0.87 | 0.835 |
| **Random Forest**       | 0.85 | 0.88 | 0.86 | 0.831 |
| **XGBoost**            | 0.85 | 0.87 | 0.86 | 0.825 |
| **Tuned Random Forest** | 0.85 | 0.89 | 0.87 | 0.829 |

📌 **Key Takeaways:**  
- The **Tuned Random Forest** has the **highest recall (0.89)**, making it the best model for detecting depression cases.  
- **Logistic Regression and Random Forest** perform similarly, with **Logistic Regression having the highest AUC-ROC (0.835)**.  
- **XGBoost performs well but slightly underperforms compared to other models.**  

---

### **Feature Importance (Random Forest)**  

| Rank | Feature | Importance Score |
|------|-------------------------------------|----------------|
| 1️⃣  | Have you ever had suicidal thoughts? | **0.210** |
| 2️⃣  | Academic Pressure                    | **0.171** |
| 3️⃣  | CGPA                                 | **0.103** |
| 4️⃣  | Financial Stress                     | **0.102** |
| 5️⃣  | Age                                  | **0.095** |
| 6️⃣  | Work/Study Hours                     | **0.081** |
| 7️⃣  | Degree                               | **0.073** |
| 8️⃣  | Study Satisfaction                   | **0.048** |
| 9️⃣  | Sleep Duration                       | **0.039** |
| 🔟  | Dietary Habits                        | **0.038** |

📌 **Insights from Feature Importance:**  
- **Suicidal Thoughts** is the most critical feature, indicating a strong link to depression.  
- **Academic Pressure & Financial Stress** play a significant role in students' mental health.  
- **Sleep Duration & Dietary Habits** also contribute but have less impact than stress-related factors.  

## Results
📊 Feature Importance Analysis (Random Forest):
1️⃣ Academic Pressure – Strong correlation with depression
2️⃣ Work Pressure – Affects mental health significantly
3️⃣ Financial Stress – Contributes to anxiety and mental disorders
4️⃣ Suicidal Thoughts – Strongest indicator
5️⃣ Family History of Mental Illness – Increases risk factor

📌 **Insights from Feature Importance:**  
- **Suicidal Thoughts** is the most critical feature, indicating a strong link to depression.  
- **Academic Pressure & Financial Stress** play a significant role in students' mental health.  
- **Sleep Duration & Dietary Habits** also contribute but have less impact than stress-related factors.  

### Ethical Considerations

⚠ Potential Risks:
Privacy Concerns: Sensitive mental health data should be anonymized.
Bias in Data: If dataset is not diverse, predictions may be biased.
False Positives/Negatives: Incorrect predictions could lead to mental health consequences.

🔍 Mitigation Strategies:
Ensuring Data Anonymity
Bias Checking before deploying the model
Collaborating with Mental Health Experts

## Future Improvements

🔮 Possible Enhancements:
Deep Learning (LSTMs for text-based depression symptoms)
SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalance
More Real-World Data to generalize across universities
Deploy as Web App (Flask/Streamlit API) for real-time predictions

### 🚀 Next Steps:
Collect more real-world mental health data.
Improve model interpretability with SHAP/LIME.
Conduct further research on causal relationships in mental health factors.
