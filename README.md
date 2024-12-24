# Chronic-Kidney-Disease-Detection

Kidney Disease Prediction
This project is aimed at predicting kidney disease using a machine learning approach. The dataset contains various medical attributes that help predict whether a patient is suffering from chronic kidney disease (CKD). The project uses various machine learning models like Support Vector Machine (SVM), Random Forest, and Gradient Boosting Machine (GBM) to predict kidney disease and compares the performance of these models.

Dataset
The dataset used in this project is kidney_disease.csv, which contains the following columns:

Age: The age of the patient
Blood Pressure: Blood pressure level
Specific Gravity: The specific gravity of urine
Albumin: Presence of albumin in urine
Sugar: Presence of sugar in urine
Red Blood Cells: Presence of red blood cells in urine
Pus Cells: Presence of pus cells in urine
Bacteria: Presence of bacteria in urine
Blood Glucose Random: Random blood glucose levels
Blood Urea: Blood urea levels
Serum Creatinine: Serum creatinine levels
Sodium: Sodium levels in the blood
Potassium: Potassium levels in the blood
Hemoglobin: Hemoglobin levels
Packed Cell Volume: Packed cell volume percentage
White Blood Cell Count: White blood cell count
Red Blood Cell Count: Red blood cell count
Hypertension: Whether the patient has hypertension
Diabetes Mellitus: Whether the patient has diabetes
Coronary Artery Disease: Whether the patient has coronary artery disease
Appetite: Whether the patient has appetite
Pedal Edema: Whether the patient has pedal edema
Anemia: Whether the patient has anemia
Classification (Class): Whether the patient is classified as having Chronic Kidney Disease (CKD) or not (0 = CKD, 1 = Not CKD)

## Project Steps
1. Data Preprocessing
The dataset was cleaned by handling missing values, replacing non-standard values, and converting columns to appropriate data types.
Columns were renamed for better readability, and the target variable (classification) was converted into binary values (0 and 1).
2. Exploratory Data Analysis (EDA)
Plots for the distribution of numeric variables were created using matplotlib and seaborn.
Categorical variable distributions were visualized using matplotlib and plotly.
3. Handling Missing Data
Numerical columns with missing values were imputed using the IterativeImputer method.
Categorical columns were imputed using random sampling for some columns and mode imputation for others.
4. Feature Engineering
Categorical variables were label encoded using LabelEncoder.
Features were standardized using StandardScaler to ensure all features were on the same scale.
5. Model Training and Evaluation
The dataset was split into training and testing sets.
Models such as Support Vector Machine (SVM), Random Forest Classifier, and Gradient Boosting Classifier were trained and evaluated.
Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix were used to evaluate the models.
6. Model Comparison
The performance of the models was compared based on various evaluation metrics such as accuracy, precision, recall, and F1-score.
Visualizations of model performance were created using plotly for a clear comparison.

## Libraries Used
NumPy: For numerical computations
Pandas: For data manipulation and analysis
Matplotlib/Seaborn: For data visualization
Plotly: For interactive visualizations
Scikit-learn: For machine learning models and metrics

## Conclusion
The project demonstrates how machine learning models can be used to predict chronic kidney disease (CKD). Various models were trained and evaluated, and their performances were compared. The best-performing model can be used to assist healthcare professionals in diagnosing CKD.

## Future Improvements
Experimenting with more advanced imputation techniques
Hyperparameter tuning for models
Implementing cross-validation for better model performance

## Getting Started
Clone the repository:
git clone https://github.com/ifeanyiojji/Kidney-Disease-Prediction.git

Install the required libraries
pip install -r requirements.txt

Run the code () in a Python environment
