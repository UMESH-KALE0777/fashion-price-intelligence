👑 Fashion Price Intelligence

AI-Powered Fashion Pricing & Insights Dashboard

🧠 Overview

Fashion Price Intelligence is a machine learning–powered dashboard that predicts fashion product prices and generates actionable business insights.
It uses AI, data visualization, and business intelligence to analyze fashion market trends, brand performance, and pricing strategies.

This project bridges machine learning and real-world retail analytics — perfect for fashion startups and pricing analysts.

⚙️ Features

✅ Upload your fashion store dataset (CSV)
✅ Automatically trains an ML model (CatBoostRegressor)
✅ Predicts product prices based on brand, category, and popularity
✅ Displays Actual vs Predicted results visually
✅ Generates premium business insights, including:
Category performance
Brand strength index
Bestseller identification
Price strategy recommendations

🧩 Tech Stack
Category	Tools / Libraries
Language	Python
ML Model	CatBoostRegressor
Data Handling	Pandas, NumPy, Scikit-Learn
Visualization	Plotly, Streamlit
Deployment	Streamlit Cloud
Version Control	Git + GitHub

📂 Project Structure
fashion_predictor_real_one/
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard UI & logic
│
├── data/
│   ├── raw_data.csv            # Sample dataset
│   ├── clean_data.csv          # Processed data
│   └── predictions.csv         # Output predictions
│
├── models/
│   └── trained_model.cbm       # Saved CatBoost model
│
├── src/                        # Core ML logic (modular structure)
│   ├── data_ingest.py          # Load and validate datasets
│   ├── data_cleaning.py        # Clean and preprocess data
│   ├── feature_engineering.py  # Create additional features
│   ├── train_model.py          # Train and save CatBoost model
│   ├── predict.py              # Generate price predictions
│   ├── evaluate.py             # Calculate R², RMSE, MAE
│   ├── visualize.py            # Plot charts with Plotly
│   ├── utils.py                # Helper functions
│   └── config.py               # Configuration constants
│
├── README.md                   # (You are here)
└── requirements.txt            # Dependencies

🚀 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/UMESH-KALE0777/fashion-price-intelligence.git
cd fashion-price-intelligence

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/Scripts/activate   # for Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run dashboard/app.py

📊 Example Output

R² Score: 0.97
Train Rows: 20,000
Test Rows: 5,000
Predictions: Highly accurate with stable variance

💎 Key Learnings

Building full ML pipelines with modular design (src/ structure).
Integrating business analytics with predictive modeling.
Deploying production-ready dashboards using Streamlit Cloud.
Improving interpretability through category-level insights.

🏷️ Tags

Machine Learning Python Data Science Streamlit CatBoost Fashion AI
Business Intelligence Retail Analytics Data Visualization
