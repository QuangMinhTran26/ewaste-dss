import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("updated_e_waste_dataset.csv")

condition_map = {'Broken': 0, 'Average': 1, 'Good': 2}
device_type_map = {'Appliance': 0, 'Consumer Electronics': 1, 'IT Equipment': 2}

df['Condition_encoded'] = df['Device Condition'].map(condition_map)
df['DeviceType_encoded'] = df['Device Type'].map(device_type_map)

metals = ['Gold (g)', 'Silver (g)', 'Aluminum (g)',
          'Platinum (g)', 'Nickel (g)', 'Tin (g)',
          'Lithium (g)', 'Rhodium (g)']

features = ['Condition_encoded', 'DeviceType_encoded',
            'Year of Manufacture', 'Device Age',
            'Market Value of Metals']

X = df[features]
y = df[metals]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

metal_prices = {
    'Gold (g)': 85.0, 'Silver (g)': 0.9, 'Aluminum (g)': 0.0025,
    'Platinum (g)': 32.0, 'Nickel (g)': 0.014, 'Tin (g)': 0.026,
    'Lithium (g)': 0.013, 'Rhodium (g)': 4.7
}

default_recovery_cost = {  #for completely new device with unknown cost of recovery
    'Consumer Electronics': 27.68, #the default cost is average of each catergoy
    'Appliance': 27.57,
    'IT Equipment': 27.43
}

recovery_rate = 0.90


st.header("E-Waste DSS — Gatekeeping Recommendation")


condition = st.selectbox("Device Condition", ['Broken', 'Average', 'Good'])
device_type = st.selectbox("Device Type", ['Appliance', 'Consumer Electronics', 'IT Equipment'])
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2019)
age = st.number_input("Device Age (years)", min_value=0, max_value=25, value=5)
market_value = st.number_input("Market Value of Metals (USD)", min_value=0.0, value=300.0)
default_cost = default_recovery_cost[device_type]
cost = st.number_input(
    f"Cost of Recovery (USD) — default for {device_type}",
    min_value=0.0,
    value=default_cost
)
if st.button("Run Analysis"):
    input_data = pd.DataFrame([{
        'Condition_encoded': condition_map[condition],
        'DeviceType_encoded': device_type_map[device_type],
        'Year of Manufacture': year,
        'Device Age': age,
        'Market Value of Metals': market_value
    }])

    predicted = knn.predict(input_data)[0]
    scenarios = {'Pessimistic': 0.80, 'Base': 1.00, 'Optimistic': 1.20}
    results = {}

    for name, multiplier in scenarios.items():
        revenue = sum(predicted[i] * metal_prices[m] * multiplier * recovery_rate
                     for i, m in enumerate(metals))
        results[name] = revenue - cost

    st.subheader("Results")
    for name, npv in results.items():
        st.write(f"**{name} NPV:** ${npv:.2f}")

    if all(v > 0 for v in results.values()):
        st.success("STRONG RECOMMENDATION: DISASSEMBLE")
    elif results['Base'] > 0:
        st.warning("CONDITIONAL RECOMMENDATION: DISASSEMBLE — monitor price risk")
    else:
        st.error("RECOMMENDATION: MANUAL CHECK")