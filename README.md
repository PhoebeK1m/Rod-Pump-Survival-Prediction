# Rod-Pump Survival Prediction  
**Presentation:**  
[▶ Click here to view the slide deck](https://onedrive.live.com/personal/08b740479a175a64/_layouts/15/Doc.aspx?sourcedoc=%7B6fddb6ed-46da-48c3-8d1b-a9dfdcd8439e%7D&action=default)

---

## Overview
This project builds a **Cox Proportional Hazards survival model** to predict failure risk for rod-pump wells.  

It produces:

- Hazard ratios  
- Survival curve predictions  
- High-risk vs low-risk comparisons  
- Feature importance visualizations  
- A CSV of model-generated risk scores  

No proprietary dataset information is shown.  
Only modeling methods and results are documented.

---

## Modeling Approach

### **1. Feature Engineering**
- Standardize numeric features  
- One-hot encode categorical fields  
- Log-transform skewed variables  
- Drop zero-variance columns  
- Create spline terms for high-variance continuous predictors  

### **2. Feature Selection — LASSO Cox Model**
- L1-penalized Cox model used to shrink/zero-out weak features  
- Top contributors + spline terms are retained  

### **3. Final Model — Ridge Cox Model**
- L2-penalized Cox model trained using selected features  
- Produces:
  - Hazard ratios  
  - Survival functions  
  - Partial hazard scores (“risk scores”)  

---

## Model Results

### **Survival Curves — Highest-Risk Wells**
Shows rapid probability decline for wells predicted to fail earliest.  
**File:** `knn/top10_high_risk_survival.png`


### **Survival Curves — Lowest-Risk Wells**
Low-risk wells maintain high survival probability over time.  
**File:** `knn/bottom10_low_risk_survival.png`


### **Feature Importance (Hazard Ratios)**
Two plots summarize the strongest predictors:

- **Risk-increasing features:**  
  `knn/top10_highest_coefficients.png`
- **Protective features:**  
  `knn/top10_lowest_coefficients.png`


### **Kaplan–Meier Risk Group Validation**
Wells are split into *High Risk* vs *Low Risk* based on model output.

**File:** `knn/Kaplan–Meier_Curves_High_vs_Low_Risk.png`  

Clear separation = strong model discrimination.

---

### **Output Risk Score File**
`knn/cox_failure_risk_scores.csv` contains:

- Model-generated risk score  
- Assigned risk group  
- Processed feature columns  

---

## How to Run

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **2. Add Your Data**

Edit the script to load your dataset:
```python
df = pd.read_csv("your_file.csv")
```

### **3. Run the Model**

Run the following command in your terminal:
```bash
python model.py
```

All results are presented without exposing any sensitive data.