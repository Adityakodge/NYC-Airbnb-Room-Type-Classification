# NYC-Airbnb-Room-Type-Classification
# 🏠 NYC Airbnb Room Type Classification

## 📌 Project Overview

This project uses **Machine Learning classification techniques** to predict the **room type** of Airbnb listings in New York City.

The model predicts one of three room types:

* `Entire home/apt`
* `Private room`
* `Shared room`

The project follows a complete **Machine Learning lifecycle**, starting from data loading and exploratory data analysis to preprocessing, model comparison, hyperparameter tuning, final evaluation, and saving the trained model pipeline.

---

## 🎯 Project Objective

The main objective is to build a machine learning model that can predict the `room_type` of an Airbnb listing using information such as:

* Location
* Price
* Minimum nights
* Number of reviews
* Reviews per month
* Availability
* Host listing count
* Latitude and longitude
* Neighbourhood
* Neighbourhood group

---

## 📊 Dataset

The project uses the **New York City Airbnb Open Data** dataset from Kaggle.

**Dataset:** New York City Airbnb Open Data

The dataset contains Airbnb listings from New York City with information about hosts, locations, prices, reviews, availability, and room types.

### Target Variable

`room_type`

The target contains three classes:

```text
Entire home/apt
Private room
Shared room
```

---

## 🔄 Machine Learning Workflow

The project follows these major steps:

```text
Data Collection
      ↓
Data Loading
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Final Evaluation
      ↓
Model Saving
```

---

## 🔍 Exploratory Data Analysis

The following EDA techniques were performed:

### Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

### Target Variable Analysis

The distribution of different room types was analyzed.

The dataset contains an **imbalanced target variable**, with `Shared room` representing a much smaller class.

### Univariate Analysis

The distributions of numerical variables were analyzed, including:

* Price
* Minimum nights
* Number of reviews
* Reviews per month
* Calculated host listing count
* Availability

### Bivariate Analysis

Relationships between features and the target variable were explored, including:

* Room type vs. price
* Room type distribution

### Correlation Analysis

Correlation between numerical variables was visualized using a heatmap.

### Geographic Analysis

Latitude and longitude were plotted to understand the geographic distribution of Airbnb room types across New York City.

---

## 🧹 Data Cleaning

The following cleaning steps were performed.

### Removed Unnecessary Columns

The following columns were removed:

```text
id
name
host_id
host_name
last_review
```

These columns were considered identifiers, free text, or unsuitable for the general tabular model.

### Handling Missing Values

Missing values in `reviews_per_month` were replaced with `0`.

```python
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
```

This represents listings with no review activity.

### Outlier Treatment

Extreme values in:

* `price`
* `minimum_nights`

were capped at the **99th percentile** rather than removing the records.

```python
price_cap = df_clean['price'].quantile(0.99)
nights_cap = df_clean['minimum_nights'].quantile(0.99)

df_clean['price'] = df_clean['price'].clip(upper=price_cap)
df_clean['minimum_nights'] = df_clean['minimum_nights'].clip(upper=nights_cap)
```

---

## ⚙️ Feature Preparation

The target variable was separated from the input features.

```python
X = df_clean.drop(columns=['room_type'])
y = df_clean['room_type']
```

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)
```

`stratify=y` was used to maintain the class proportions between the training and testing datasets.

---

## 🔧 Data Preprocessing

A `ColumnTransformer` and `Pipeline` were used to create a consistent preprocessing workflow.

### Numerical Features

Numerical features were processed using:

1. Median imputation
2. Standard scaling

```text
Median Imputation
       ↓
Standard Scaling
```

### Categorical Features

Categorical features were processed using:

1. Most-frequent imputation
2. One-hot encoding

```text
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

This preprocessing was included inside the model pipeline to reduce the risk of data leakage.

---

## 🤖 Machine Learning Models

Four classification algorithms were compared:

### 1. Logistic Regression

Used as a simple and interpretable baseline.

### 2. Decision Tree

Used to capture non-linear relationships between features.

### 3. Random Forest

An ensemble learning algorithm using multiple decision trees.

### 4. Gradient Boosting

A sequential ensemble method that builds models to improve previous errors.

---

## ⚖️ Handling Class Imbalance

Because the target variable is imbalanced, `class_weight="balanced"` was used for:

* Logistic Regression
* Decision Tree
* Random Forest

The project used **Macro F1-score** along with accuracy because macro F1 gives equal importance to each class.

---

## 📈 Model Comparison

Each model was evaluated using **3-fold stratified cross-validation**.

Two evaluation metrics were used:

* Accuracy
* Macro F1-score

Macro F1 was particularly important because the dataset contains an imbalanced target variable.

---

## 🎛️ Hyperparameter Tuning

Random Forest was used for hyperparameter tuning with `RandomizedSearchCV`.

The parameters explored included:

```python
n_estimators
max_depth
min_samples_split
```

Example search space:

```python
param_distribution = {
    "classifier__n_estimators": [100, 200, 150, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10]
}
```

The model was optimized using:

```text
Macro F1-score
```

with 3-fold cross-validation.

---

## 🧪 Final Model Evaluation

After hyperparameter tuning, the best pipeline was evaluated on the previously untouched test dataset.

The following metrics were calculated:

### Accuracy

Measures the percentage of correctly classified listings.

### Macro F1-score

Calculates the F1-score independently for each class and then takes their average.

### Confusion Matrix

A confusion matrix was created to analyze how well the model classified each room type.

---

## 💾 Model Saving

The complete preprocessing and machine learning pipeline was saved as:

```text
Model_Pipeline.pkl
```

The pipeline was serialized using `joblib`.

```python
import joblib

joblib.dump(
    best_pipeline,
    "Model_Pipeline.pkl",
    compress=3
)
```

This allows the preprocessing and trained model to be reused together for future predictions.

---

## 🛠️ Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Programming language      |
| Pandas           | Data manipulation         |
| NumPy            | Numerical operations      |
| Matplotlib       | Data visualization        |
| Seaborn          | Statistical visualization |
| Scikit-learn     | Machine Learning          |
| Joblib           | Model serialization       |
| Kaggle           | Dataset source            |
| Jupyter Notebook | Development environment   |

---

## 📁 Project Structure

```text
NYC-Airbnb-Room-Type-Classification/
│
├── nyc_airbnb_room_type_classification.ipynb
│
├── Model_Pipeline.pkl
│
├── README.md
│
└── requirements.txt
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/NYC-Airbnb-Room-Type-Classification.git
```

### 2. Navigate to the Project

```bash
cd NYC-Airbnb-Room-Type-Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib kagglehub jupyter
```

### 6. Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
nyc_airbnb_room_type_classification.ipynb
```

and run the cells sequentially.

---

## 📌 Key Learning Outcomes

Through this project, I worked on:

* Exploratory Data Analysis
* Missing value handling
* Outlier treatment
* Feature engineering
* Handling imbalanced classification data
* Train-test splitting
* Data preprocessing
* `ColumnTransformer`
* Machine Learning pipelines
* Classification algorithms
* Cross-validation
* Model comparison
* Hyperparameter tuning
* Model evaluation
* Confusion matrix
* Model serialization

---

## 🚀 Future Improvements

Possible future improvements include:

* Deploying the model using **Streamlit or Flask**
* Creating an interactive prediction interface
* Performing more extensive feature engineering
* Testing additional classification algorithms
* Applying more advanced imbalance-handling techniques
* Adding automated model monitoring
* Creating a complete production-ready ML API

---

## 👨‍💻 Author

**Aditya Kodge**

PGDM Student | Data Science & Business Analytics

### Areas of Interest

* Data Science
* Machine Learning
* Data Analytics
* Business Analytics
* Python
* SQL
* Power BI

---

## ⭐ Project Summary

This project demonstrates an end-to-end **Machine Learning classification workflow** using the New York City Airbnb dataset. It covers data exploration, cleaning, preprocessing, model comparison, hyperparameter tuning, evaluation, and model deployment preparation through a saved ML pipeline.
