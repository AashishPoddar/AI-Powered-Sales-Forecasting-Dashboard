# 📊 AI-Powered Sales Forecasting Dashboard

An end-to-end **Machine Learning + Business Intelligence** project that forecasts future sales using historical data and presents insights through an interactive **Power BI dashboard**.  
This project is built using the **Rossmann Store Sales dataset**(Kaggle), leveraging **Python, XGBoost**, and **Power BI** to deliver actionable business insights.

---

## 🚀 Project Overview

The goal of this project is to:
- Predict **future sales trends** using machine learning
- Compare **actual vs forecasted sales**
- Identify **best and worst performing months**
- Enable **interactive filtering** for business decision-making
- Present insights through a **professional Power BI dashboard**

This project simulates a **real-world analytics workflow** commonly used in retail analytics and consulting roles.

---

## 🛠️ Tech Stack

### 🔹 Programming & ML
- Python
- Pandas, NumPy
- XGBoost (Regression)
- argparse (CLI support)

### 🔹 Visualization
- Power BI Desktop

### 🔹 Version Control
- Git & GitHub

---


---

## ⚙️ Workflow Explained

### 1️⃣ Data Preprocessing
- Merged store and sales data
- Handled missing values
- Encoded categorical variables
- Engineered time-based features (Year, Month, Week, Day)
- Saved cleaned dataset for modeling

📄 Script: `src/rossmann_preprocess.py`

---

### 2️⃣ Machine Learning Model
- Aggregated daily total sales
- Created lag features and rolling averages
- Trained an **XGBoost regression model**
- Generated **14-day future sales forecast**
- Exported forecast results for Power BI

📄 Script: `src/rossmann_train_xgb.py`

Run example:
```bash
python src/rossmann_train_xgb.py --horizon 14 --rounds 200

---

### 3️⃣ Power BI Dashboard

The Power BI dashboard provides an interactive and business-focused view of historical and forecasted sales data.

---

#### 🔑 KPI Cards
- **Total Actual Sales**
- **Total Forecast**
- **Best Month**
- **Worst Month**
- **Next 14-Day Forecast Total**
- **Sales Variance %**

These KPI cards help stakeholders quickly understand overall performance and short-term expectations.

---

#### 📈 Visualizations
- **Actual vs Forecasted Sales Trend (Monthly)**  
  Line chart comparing historical sales with model predictions.
  
- **Monthly Sales Comparison (Bar Chart)**  
  Month-wise comparison of sales to identify trends and seasonality.
  
- **Yearly Sales Summary**  
  Aggregated yearly sales to compare overall performance across years.
  
- **Store-wise Sales Performance**  
  Horizontal bar chart showing sales distribution across stores to highlight top-performing stores.

---

#### 🎛️ Interactive Filters
- **Year**
- **Store**
- **Promo**
- **Store Type**

All visuals are interconnected, allowing dynamic filtering and deeper analysis.

---

#### 📁 Dashboard Files
- Power BI file:  
  `dashboard/sales_forecast_dashboard.pbix`

- Dashboard screenshots:  
  Available in the `dashboard/` folder

---

## ✅ Project Requirements Fulfilled

✔ Sales trend line with actual vs forecasted data  
✔ Monthly & yearly comparisons  
✔ Filters by store, promo, and store type  
✔ Highlighting top-performing and low-performing periods  
✔ Insight cards for business decision-making  

---

## 📦 Installation & Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


