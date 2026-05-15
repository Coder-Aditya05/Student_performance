# 🎓 Student Performance Predictor

An end-to-end Machine Learning project that predicts a student's **math score** based on demographic and academic factors. Built with a modular pipeline, EDA notebooks, and deployed as a Flask web application on AWS Elastic Beanstalk.

---

## 📌 Overview

Understanding what factors influence student performance can help educators intervene early. This project trains a regression model on student data and serves predictions through a clean web interface where users can input student details and get an instant score prediction.

---

## 🏗️ Project Structure

```
Student_performance/
├── src/
│   ├── components/          # Data ingestion, transformation, model trainer
│   ├── pipeline/            # Training & prediction pipelines
│   ├── exception/           # Custom exception handling
│   └── logger/              # Custom logging
├── notebook/                # EDA and model training notebooks
├── artifacts/               # Saved model & preprocessor
├── templates/               # HTML templates (Flask/Jinja2)
├── .ebextensions/           # AWS Elastic Beanstalk config
├── app.py                   # Flask application
├── requirements.txt
└── setup.py
```

---

## 🔍 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| Gender | Male / Female |
| Race/Ethnicity | Group A–E |
| Parental Level of Education | e.g. bachelor's degree, some college |
| Lunch | Standard / Free or reduced |
| Test Preparation Course | Completed / None |
| Reading Score | Numeric score |
| Writing Score | Numeric score |

**Target:** Math Score (continuous)

---

## ⚙️ ML Pipeline

1. **Data Ingestion** — Loads and splits data into train/test sets
2. **Data Transformation** — Applies StandardScaler for numerical features and OneHotEncoder for categorical features via a ColumnTransformer
3. **Model Training** — Trains and evaluates multiple models, selects the best performer

**Models evaluated:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, **XGBoost**, **CatBoost**, AdaBoost, Gradient Boosting

---

## 🌐 Web Application

A Flask-based UI where users fill in student details via a form and receive a predicted math score instantly.

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/predictdata` | GET | Render prediction form |
| `/predictdata` | POST | Submit form and get prediction |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| ML | scikit-learn, XGBoost, CatBoost |
| EDA | pandas, NumPy, Matplotlib, Seaborn |
| Web Framework | Flask |
| Deployment | AWS Elastic Beanstalk |

---

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Coder-Aditya05/Student_performance.git
cd Student_performance
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Visit `http://localhost:5000` to use the prediction form.

---

## ☁️ Deployment

This project is configured for deployment on **AWS Elastic Beanstalk** via the `.ebextensions` directory. The `application.py` file serves as the entry point for the Beanstalk environment.

---

## 📊 EDA Highlights

Exploratory analysis was conducted in Jupyter Notebooks covering:
- Score distributions across gender and race/ethnicity groups
- Impact of parental education and test preparation on performance
- Correlation between reading, writing, and math scores

---

## 📬 Contact

**Aditya** — [GitHub](https://github.com/Coder-Aditya05)
