# Household-Energy-Forecast

This project provides an interactive **Streamlit dashboard** and backend energy forecasting tools using machine learning and deep learning models. It focuses on **individual household energy prediction and smart grid supply management**, built on the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

---
## 🔍 Project Overview
### ✅ Data Source:
- 📥 [Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- Format: `.zip` file containing a `.txt` file
- ⚠️ After downloading, **extract the text file from the ZIP archive** before proceeding
---
## 🧾 Data Conversion
### 📄 Script: `txt_to_csv.py`
Converts the original semicolon-separated `.txt` file to a clean `.csv` file using:
```bash
python txt_to_csv.py
```
---
## ⚙️ Preprocessing & Modeling
### 📓 Notebook: `main.ipynb`
#### ✅ Tasks Performed:
- Data cleaning, parsing datetime index
- **Feature Engineering**:
  - Time-based features: `Day`, `Month`, `Year`, `Hour`, `Weekday`
  - Cyclical encoding: `sin/cos` transformations for time values
  - Indicators: `Season`, `Is_Weekend`, `Is_Peak_Hour`
- Missing value handling
- Type conversions
---
### 🔍 Modeling:
- **Machine Learning**:
  - `XGBoost`, `DecisionTree`, `RandomForest`
- **Deep Learning**:
  - `Keras` with `Dense` layers
  - `ReLU` activation
  - `Adam` optimizer
- Separate models trained for:
  - `Global_active_power`
  - `Global_apparent_power`
📁 **Models saved** in `.h5` format (Keras)
---
## 🔌 Real-World Input Challenge: Global Intensity
### ⚠️ Problem:
Users **cannot manually enter** `Global_intensity` (electric current) for each hour in the dashboard.
### ✅ Solution:
Generated **synthetic current values** using historical data.
- Calculated average intensity per:
  - `(Month, Day, Hour)`
- Exported as:
  - `mean_intensity_by_time.csv`
  - `max_intensity_by_time.csv`
  - `min_intensity_by_time.csv`
Used in the Streamlit app to simulate realistic current patterns.
---
## 📊 Streamlit Dashboard: `app.py`
### 🔧 Key Features
#### 1️⃣ Household Energy Usage Prediction
- Predicts hourly **Global Active Power** (in kWh)
- Displays:
  - Total energy usage in selected prediction period
  - Estimated billing based on:
    - 🇫🇷 **France (€)**
    - 🇮🇳 **India (₹)**
    - 🇺🇸 **USA ($)**
- 🧾 Billing table is **editable** for users to update current tariffs
- Supports flexible durations:
  - From 6-hour forecasts to full 2-month billing cycles
---
#### 2️⃣ Grid Energy Prediction for Supply
- Predicts both:
  - **Active Power** (kW)
  - **Apparent Power** (kVA)
- Computes:
  - **Power Factor**
  - **Phase Angle**
- ⚡ Voltage assumed fixed at **240V**
- ⚠️ **Warning if power factor drops below 0.85**, with suggestions for reactive power correction
---
#### 3️⃣ Smart Intensity Handling
User can choose how the current (`Global_intensity`) is calculated:
- `Mean` per (Month, Day, Hour)
- `Max` value
- `Min` value
- Mixed or custom-intensity approximations
---
## 💻 How to Run the Project
### 🧪 Step 1: Set up a virtual environment\
```bash
python -m venv venv
```
## ▶️ Step 2: Activate the environment
### Windows:
```bash
.\venv\Scripts\activate
```
### macOS/Linux:
```bash
source venv/bin/activate
```
## 📦 Step 3: Install dependencies
```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn xgboost tensorflow
```
## 🚀 Step 4: Launch the app
```bash
streamlit run app.py
```
