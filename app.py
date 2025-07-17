import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import math

# ------------------------------
# Load pre-trained models
# ------------------------------
model_active = load_model("global_active_power_model.h5", compile=False)
model_apparent = load_model("apparent_power_model.h5", compile=False)

# Load precomputed intensity values
mean_intensity = pd.read_csv("mean_intensity_by_time.csv")
min_intensity = pd.read_csv("min_intensity_by_time.csv")
max_intensity = pd.read_csv("max_intensity_by_time.csv")

# ------------------------------
# Functions
# ------------------------------
def generate_datetime_range(start, end):
    return pd.date_range(start=start, end=end, freq='H')

def preprocess_datetime(df):
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Weekday'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

    def get_season(month):
        return 0 if month in [12, 1, 2] else 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3
    df['Season'] = df['Month'].apply(get_season)

    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda h: 1 if 8 <= h <= 20 else 0)
    return df

def calculate_billing(total_kwh, france_rate, india_slabs, us_rate):
    france_cost = total_kwh * france_rate

    if total_kwh <= 100:
        india_cost = total_kwh * india_slabs[0]
    elif total_kwh <= 200:
        india_cost = 100 * india_slabs[0] + (total_kwh - 100) * india_slabs[1]
    elif total_kwh <= 500:
        india_cost = 100 * india_slabs[0] + 100 * india_slabs[1] + (total_kwh - 200) * india_slabs[2]
    else:
        india_cost = 100 * india_slabs[0] + 100 * india_slabs[1] + 300 * india_slabs[2] + (total_kwh - 500) * india_slabs[3]

    us_cost = total_kwh * us_rate
    return france_cost, india_cost, us_cost

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔌 Energy Prediction & Billing App")
st.markdown("Enter the date and hour range (00 to 23). You can customize global intensity (current) source and billing rates.")

col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input("From Date", datetime.date(2025, 5, 1))
    from_hour = st.number_input("From Hour (0-23)", min_value=0, max_value=23, value=0)
with col2:
    to_date = st.date_input("To Date", datetime.date(2025, 7, 1))
    to_hour = st.number_input("To Hour (0-23)", min_value=0, max_value=23, value=23)

option = st.radio("Select Prediction Type", ["Household Energy Usage Prediction", "Grid Energy Prediction for Supply"])

intensity_choice = st.selectbox("Select Input Current (Global Intensity)", [
    "Mean Current", "Max Current", "Min Current", "Mixed Current"
], help="This feature lets you choose what current values to use for prediction. Your model was trained using current as a feature.")

if intensity_choice == "Mixed Current":
    st.markdown("### 🎛️ Mixed Current Distribution (must total 100%)")
    col1, col2, col3 = st.columns(3)
    with col1:
        mean_pct = st.slider("% Mean", 0, 100, 80)
    with col2:
        max_pct = st.slider("% Max", 0, 100, 15)
    with col3:
        min_pct = st.slider("% Min", 0, 100, 5)
    total_pct = mean_pct + max_pct + min_pct
    if total_pct != 100:
        st.error("The percentages must sum up to 100%.")

if option == "Household Energy Usage Prediction":
    st.markdown("---")
    st.markdown("### 💵 Billing Rates")
    with st.expander("Update Regional Billing Rates"):
        france_rate = st.number_input("France rate (€/kWh)", value=0.2276, help="This is the last updated billing amount.")
        us_rate = st.number_input("US rate ($/kWh)", value=0.18, help="This is the last updated billing amount.")
        india_rate_0 = st.number_input("India 0–100 units (₹/kWh)", value=3.5)
        india_rate_1 = st.number_input("India 101–200 units (₹/kWh)", value=4.6)
        india_rate_2 = st.number_input("India 201–500 units (₹/kWh)", value=6.6)
        india_rate_3 = st.number_input("India >500 units (₹/kWh)", value=7.1)
        confirm_rates = st.button("OK to Confirm Updated Billing Rates")
else:
    confirm_rates = True

# Store billing region selection persistently
if "billing_region" not in st.session_state:
    st.session_state.billing_region = "France"
if option == "Household Energy Usage Prediction":
    st.session_state.billing_region = st.radio(
        "Select Region for Billing",
        ["France", "India", "USA"],
        horizontal=True,
        index=["France", "India", "USA"].index(st.session_state.billing_region)
    )

if st.button("Run Prediction") and (intensity_choice != "Mixed Current" or total_pct == 100):
    start = pd.Timestamp(f"{from_date} {from_hour}:00")
    end = pd.Timestamp(f"{to_date} {to_hour}:00")
    df = pd.DataFrame(index=generate_datetime_range(start, end))
    df = preprocess_datetime(df)

    base = df.reset_index().rename(columns={'index': 'Datetime'})
    base['Hour'] = base['Hour'].astype(int)
    base['Day'] = base['Day'].astype(int)
    base['Month'] = base['Month'].astype(int)

    if intensity_choice == "Mean Current":
        df_merged = pd.merge(base, mean_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df['Global_intensity'] = df_merged['Mean_Global_Intensity'].values
    elif intensity_choice == "Max Current":
        df_merged = pd.merge(base, max_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df['Global_intensity'] = df_merged['Max_Global_Intensity'].values
    elif intensity_choice == "Min Current":
        df_merged = pd.merge(base, min_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df['Global_intensity'] = df_merged['Min_Global_Intensity'].values
    elif intensity_choice == "Mixed Current":
        df_mean = pd.merge(base, mean_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df_max = pd.merge(base, max_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df_min = pd.merge(base, min_intensity, on=['Month', 'Day', 'Hour'], how='left')
        df['Global_intensity'] = (
            df_mean['Mean_Global_Intensity'] * (mean_pct / 100) +
            df_max['Max_Global_Intensity'] * (max_pct / 100) +
            df_min['Min_Global_Intensity'] * (min_pct / 100)
        )

    feature_cols = ['Global_intensity', 'Day', 'Month', 'Year', 'Weekday', 'Hour', 'Is_Weekend', 'Season',
                    'Hour_sin', 'Hour_cos', 'Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos', 'Is_Peak_Hour']

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    if option == "Household Energy Usage Prediction":
        y_pred = model_active.predict(X).flatten()
        df['Predicted_Active_Power_kW'] = y_pred
        df['Energy_kWh'] = df['Predicted_Active_Power_kW']
        total_energy = df['Energy_kWh'].sum()

        st.subheader("🔋 Total Predicted Energy Usage")
        st.metric("Total Energy (kWh)", f"{total_energy:.2f} kWh")

        billing_region = st.session_state.billing_region

        if billing_region == "France":
            amount = total_energy * france_rate
            st.success(f"🇫🇷 France Billing: €{amount:.2f}")
        elif billing_region == "USA":
            amount = total_energy * us_rate
            st.success(f"🇺🇸 USA Billing: ${amount:.2f}")
        else:
            if total_energy <= 100:
                amount = total_energy * india_rate_0
            elif total_energy <= 200:
                amount = 100 * india_rate_0 + (total_energy - 100) * india_rate_1
            elif total_energy <= 500:
                amount = 100 * india_rate_0 + 100 * india_rate_1 + (total_energy - 200) * india_rate_2
            else:
                amount = (
                    100 * india_rate_0 +
                    100 * india_rate_1 +
                    300 * india_rate_2 +
                    (total_energy - 500) * india_rate_3
                )
            st.success(f"🇮🇳 India Billing: ₹{amount:.2f}")

        if confirm_rates:
            france_cost, india_cost, us_cost = calculate_billing(
                total_energy, france_rate, [india_rate_0, india_rate_1, india_rate_2, india_rate_3], us_rate)
            st.subheader("💰 Billing Estimates")
            billing_df = pd.DataFrame({
                'Region': ['France', 'India', 'USA'],
                'Estimated Cost': [f"€{france_cost:.2f}", f"₹{india_cost:.2f}", f"${us_cost:.2f}"]
            })
            st.dataframe(billing_df, use_container_width=True)

    else:
        y_active = model_active.predict(X).flatten()
        y_apparent = model_apparent.predict(X).flatten()

        df['Active_kW'] = y_active
        df['Apparent_kVA'] = y_apparent
        df['Power_Factor'] = df['Active_kW'] / df['Apparent_kVA']
        df['Power_Factor'] = df['Power_Factor'].clip(0, 1)
        df['Phase_Angle_deg'] = df['Power_Factor'].apply(lambda pf: math.degrees(math.acos(pf)))

        total_active = df['Active_kW'].sum()
        total_apparent = df['Apparent_kVA'].sum()
        avg_pf = df['Power_Factor'].mean()

        st.subheader("📊 Grid Power Summary")
        st.metric("Total Active Power (kW)", f"{total_active:.2f} kW")
        st.metric("Total Apparent Power (kVA)", f"{total_apparent:.2f} kVA")
        st.metric("Average Power Factor", f"{avg_pf:.3f}")

        if avg_pf >= 0.95:
            st.success("✅ Power factor is excellent and within acceptable range.")
        elif avg_pf >= 0.85:
            st.warning("⚠️ Power factor is moderate. Consider correction if load increases.")
        else:
            st.error("❗ Low power factor. Consider power factor correction.")