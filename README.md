# ML-Project

# HR Employee Attrition Predictor

I built this project to practice end-to-end machine learning on a real-world HR problem — predicting whether an employee is likely to leave a company based on their profile.

The motivation is simple: losing an employee costs a company between 50% and 200% of their annual salary. If HR can identify at-risk employees early, they can act before it's too late.

# The Data

Synthetic dataset modeled after IBM's HR Analytics dataset — 1,470 employees, 31 columns covering age, salary, department, job satisfaction, overtime status, years at company, and more.

Target column: Attrition — did the employee leave? Yes or No.
Class split: 81% stayed, 19% left.

# What I Did

Preprocessing — separated numeric and categorical features, handled missing values, encoded the target to 0/1, split 70/30 train/test, applied StandardScaler on numeric columns and OneHotEncoder on categorical ones.

Feature Engineering — created 3 custom features that ended up being highly predictive:
- Income_Per_Year_Experience — salary relative to years of experience. Ended up being the #1 most important feature in the model.
- Tenure_Ratio — how long the employee stayed at this company vs their total career length.
- OverTime_Satisfaction — a combined risk score: working overtime with low job satisfaction is a strong signal for attrition.

Models trained:
- Logistic Regression — 81.2%
- Random Forest — 80.9%
- XGBoost — 78.7%
- Random Forest after GridSearchCV tuning — 81.9% (best)

# Python, scikit-learn, XGBoost, pandas, matplotlib, seaborn
