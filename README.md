# AutoML Streamlit Pipeline

An end-to-end AutoML application built using **Streamlit** and **FLAML**.

## Features
- Upload CSV datasets
- Automated EDA using ydata-profiling
- AutoML regression using FLAML
- Model comparison and selection
- Download trained model (.pkl)

## Tech Stack
- Python 3.10
- Streamlit
- FLAML
- Scikit-learn
- Pandas
- ydata-profiling

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
