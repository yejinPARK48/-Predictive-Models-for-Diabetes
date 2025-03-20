# 🩸 Diabetes Prediction Analysis  
This study delves into the potential correlation between the wealth gap, education level, and diabetes beyond lifestyle habits and health conditions.  

Survey data, consisting of **22 distinct categories**, was collected from **253,680 Americans** through the **BRFSS (Behavioral Risk Factor Surveillance System)**.  
Using **R**, I employed **Logistic Regression, Decision Tree, and Artificial Neural Networks** to establish predictive models for diabetes.  

---

## 📊 1. Data Overview
The dataset contains **both categorical and continuous variables**(N=253680), summarized in the following table:  

### Nominal Variables

| Symbol | Variable | Description |
|--------|----------|------------|
| **Y**  | Diabetes_binary | Diabetes (0 = No, 1 = Yes) |
| **X1** | HighBP | Blood Pressure (0 = Normal, 1 = High) |
| **X2** | HighChol | Cholesterol (0 = Normal, 1 = High) |
| **X3** | CholCheck | Cholesterol Check in the Past 5 Years (0 = No, 1 = Yes) |
| **X4** | Smoker | Ever smoked at least 100 cigarettes in lifetime (0 = No, 1 = Yes) |
| **X5** | Stroke | History of Stroke (0 = No, 1 = Yes) |
| **X6** | HeartDiseaseorAttack | Coronary Heart Disease or Myocardial Infarction (0 = No, 1 = Yes) |
| **X7** | PhysActivity | Physical Activity in the Last 30 Days (excluding work) (0 = No, 1 = Yes) |
| **X8** | Fruits | Consumes fruit at least once a day (0 = No, 1 = Yes) |
| **X9** | Veggies | Consumes vegetables at least once a day (0 = No, 1 = Yes) |
| **X10** | HvyAlcoholConsump | Heavy Alcohol Consumption (Men: ≥14 drinks/week, Women: ≥7 drinks/week) (0 = No, 1 = Yes) |
| **X11** | AnyHealthcare | Has Health Insurance (0 = No, 1 = Yes) |
| **X12** | NoDocbcCost | Unable to see a doctor in the past 12 months due to cost (0 = No, 1 = Yes) |
| **X13** | DiffWalk | Difficulty Walking or Climbing Stairs (0 = No, 1 = Yes) |
| **X14** | Sex | Sex (0 = Female, 1 = Male) |

---

### Continuous & Ordinal Variables

| Symbol | Variable | Description |
|--------|----------|------------|
| **X15** | BMI | Body Mass Index (Continuous) |
| **X16** | Age | Age Group (1 = 18-29, 2 = 30-39, 3 = 40-49, 4 = 50-59, 5 = 60-69, 6 = 70-79, 7 = 80+) |
| **X17** | Income | Annual Income Level (1 = ≤$10,000, 2 = $10,000-$15,000, 3 = $15,000-$20,000, 4 = $20,000-$25,000, 5 = $25,000-$35,000, 6 = $35,000-$50,000, 7 = $50,000-$75,000, 8 = >$75,000) |
| **X18** | MentHlth | Number of days in the past 30 days with poor mental health (Range: 1-30) |
| **X19** | PhysHlth | Number of days in the past 30 days with poor physical health (Range: 1-30) |
| **X20** | GenHlth | General Health Status (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor) |
| **X21** | Education | Education Level (1 = Kindergarten only, 2 = Grades 1-8, 3 = Grades 9-11, 4 = High School Graduate, 5 = Some College (1-3 years), 6 = College Graduate (4+ years)) |

---

## ⚙️ 2. Methodology  

- **📌 Logistic Regression:** Used as a baseline model, identifying the **top 5 most important variables** for diabetes prediction. The variable importance plot clearly shows a steep drop in importance beyond the top 5, justifying the feature selection.  

- **🌳 Decision Tree:** The initial model misclassified all cases as non-diabetic. However, when refined to use **only the top 5 most important variables**, the model's **performance significantly improved**, achieving a **high AUC score**. This refined approach aligns with the findings from logistic regression.  

- **🧠 Artificial Neural Network (ANN):** Attempted classification using both the full dataset and the top variables. However, **the ANN model failed to achieve meaningful classification performance**, leading to its exclusion from further analysis.

---

## 📈 3. Model Performance  

### ✅ Logistic Regression  
- The logistic regression model identified the **top 5 most significant variables**. 
- The model equation is as follows:  

$$ y = -2.99 + 0.43 \times \text{BMI} + 0.62 \times \text{GenHlth} + 0.80 \times \text{HighBP} + 0.44 \times \text{Age} + 0.59 \times \text{HighChol} $$

- **Feature Importance Plot:**
<p align='center'>
  <img src='https://github.com/user-attachments/assets/727756d9-916f-4427-aeb7-23cca555df2f'
    </p>
  
---

### 🌳 Decision Tree  
- Initially, the decision tree model **misclassified all cases as non-diabetic** due to noise from low-importance features.  
- After refining the model to **use only the top 5 most important variables**, classification accuracy **significantly improved**.  

- **Feature Importance Plot:**  
<p align='center'>
  <img src='https://github.com/user-attachments/assets/0282d42f-4271-46bd-b0e6-f36914f2ca25'>
</p>

- **Final Decision Tree Model:**
<p align='center'>
  <img src='https://github.com/user-attachments/assets/b464caab-93f2-45bb-ad1a-f17c9539be0d'>
</p>

---

### 🧠 Artificial Neural Network (ANN)  
- Due to the large dataset, a **random sample of 10,000 records** was extracted for ANN modeling.
  
- Two ANN models were tested:  
  1️⃣ **ANN with all variables**  
  2️⃣ **ANN with only the top 5 most important variables**  

#### 📌 **Findings**  
- Both models **classified nearly all data as non-diabetic (0)**, making them useless for actual prediction.  
- **Sensitivity was close to 0**, meaning the model completely failed to identify diabetic cases.  
- Despite an **AUC score of 0.83**, the misclassification issue rendered the ANN model ineffective.  

#### 📊 **Confusion Matrices**  
| Model | Predicted 0 | Predicted 1 |
|--------|-------------|-------------|
| **All Variables ANN** | **3442**  | **554**  |
|  | **2**  | **2**  |
| **5 Important Variables ANN** | **3444**  | **556**  |
|  | **0**  | **0**  |

## 🏁 6. Conclusion  

This study explored the relationship between diabetes and various factors, including physical health indicators and socio-economic conditions.  

### ✅ **Key Findings**  
- **Self-assessed health score** emerged as the **most significant predictor** across all models.  
- **Income showed a strong correlation with self-assessed health score**, suggesting that socioeconomic status plays a major role in perceived health and, consequently, diabetes risk.   

### 🏆 **Best Model: Logistic Regression**  
- **AUC = 0.81**, making it the most reliable predictive model.  
**Decision Tree (AUC = 0.74) performed reasonably well**, but its performance was lower compared to Logistic Regression.  
- **ANN was ineffective**, as it misclassified nearly all cases as non-diabetic.  

### 🔥 **Final Conclusion**  
Diabetes onset is influenced by a **complex interplay of physical health factors (e.g. self-assessed health score) and socioeconomic conditions (e.g., income)**.  
This finding highlights the importance of **considering both medical and socio-economic determinants** when understanding and addressing diabetes risk.  


















