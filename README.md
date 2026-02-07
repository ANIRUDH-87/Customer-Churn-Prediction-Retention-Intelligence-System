An end-to-end machine learning project that predicts customer churn and supports cost-aware retention decisions by combining churn risk with customer value.
The solution is deployed as an interactive Streamlit web application.

Key Features

Customer churn prediction using supervised machine learning

Churn risk segmentation (Low / Medium / High)

Business-oriented feature engineering

Retention decision logic based on customer value

Deployed as a production-ready web app

Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

Project Structure
Churn_prediction_app/
├── app.py
├── data/telco_churn_cleaned.csv
├── model/final_churn_model.pkl
├── requirements.txt

Run Locally
pip install -r requirements.txt
streamlit run Churn_prediction_app/app.py

Author

Venkata Sai Anirudh
Data Science Intern
