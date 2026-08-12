# Email Chat Ticket Routing - Execution Guide

This document explains how to run the application step-by-step, the required input/output for each step, and how to use the FastAPI service and Swagger UI.

## 1. Setup

### Command
```bash
python -m pip install pandas scikit-learn nltk joblib fastapi uvicorn tqdm pytest
```

### Required input
- Python installed on the system
- Access to the repository files in `Email_Chat_Ticket_Routing`

### Output
- Required Python packages installed
- Environment ready to run the scripts

---

## 2. Exploratory Data Analysis

### Command
```bash
python 01_eda.py
```

### Required input
- `data/customer_support_tickets.csv` file in the repo

### Output
- Printed dataset shape
- Printed column dtypes
- Printed missing-value counts
- Printed normalized distributions for `Ticket_Type` and `Ticket_Priority`
- Printed average body length in characters and words

---

## 3. Text Cleaning

### Command
```bash
python 02_text_cleaning.py
```

### Required input
- `data/customer_support_tickets.csv`

### Output
- `data/cleaned_tickets.csv`
- Printed sample raw vs cleaned text

---

## 4. Train Department Classifier

### Command
```bash
python 03_train_department_classifier.py
```

### Required input
- `data/cleaned_tickets.csv`

### Output
- `department_model.pkl`
- Printed test accuracy and macro-F1
- Printed classification report for departmental prediction

---

## 5. Train Priority Scoring Model

### Command
```bash
python 04_priority_scoring.py
```

### Required input
- `data/cleaned_tickets.csv`

### Output
- `priority_model.pkl`
- `data/priority_predictions.csv`
- Printed accuracy and confusion matrix comparing predicted priority to ground truth

---

## 6. Train Multi-label Department Model

### Command
```bash
python 05_multilabel_departments.py
```

### Required input
- `data/cleaned_tickets.csv`

### Output
- `multilabel_model.pkl`
- Printed count and percentage of test tickets flagged as multi-department
- Printed sample assigned departments with probabilities

---

## 7. Batch Routing New Tickets

### Command
```bash
python batch_score.py
```

### Optional command with custom CSV
```bash
python batch_score.py path\to\your_tickets.csv
```

### Required input
- `data/new_tickets.csv` by default, or a custom CSV path
- The input CSV must contain columns `subject` and `body`

### Output
- `data/routed_tickets.csv`
- Printed summary count of tickets routed per department

---

## 8. Run the FastAPI App

### Command
```bash
uvicorn api:app --reload
```

### Required input
- `department_model.pkl` (preferred)
- `priority_model.pkl` (preferred)
- `multilabel_model.pkl` (preferred)
- `routing.py` and `api.py`

### Output
- FastAPI service running at `http://127.0.0.1:8000`

---

## 9. API and Swagger Endpoints

### POST `/route`
- Endpoint: `http://127.0.0.1:8000/route`
- Method: `POST`
- Body: JSON
```json
{
  "subject": "Your ticket subject",
  "body": "Your ticket body text"
}
```
- Output: JSON containing
  - `clean_text`
  - `assigned_departments`
  - `priority`
  - `routing_reason`

### Swagger UI
- Open in your browser: `http://127.0.0.1:8000/docs`
- Use this page to inspect the `/route` request schema and send test requests interactively.

### Redoc UI
- Open in your browser: `http://127.0.0.1:8000/redoc`
- Use this page for an alternate API documentation view.

---

## 10. Run Tests

### Command
```bash
pytest -q
```

### Required input
- `test_routing.py` file
- Installed `pytest`

### Output
- Validation that the routing and cleaning functions work correctly

---

## Notes for new users
- Always run `02_text_cleaning.py` before training models.
- If you see a missing NLTK stopwords error, run `02_text_cleaning.py` again; it downloads the data automatically.
- Use `batch_score.py` for bulk CSV routing and `api.py` for online routing.
- The default `data/new_tickets.csv` is provided for sample input.
