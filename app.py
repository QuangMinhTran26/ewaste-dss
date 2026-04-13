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
recovery_rate = 0.90