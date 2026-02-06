# AutoML Streamlit Pipeline
Demo Link : https://automl-app-pipeline-4glc9te8kzdvxj9wrpdv5w.streamlit.app/

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
