import pandas as pd
df = pd.read_csv("updated_e_waste_dataset.csv")

# Encode Device Condition
condition_map = {'Broken': 0, 'Average': 1, 'Good': 2}
df['Condition_encoded'] = df['Device Condition'].map(condition_map)

# Encode Device Type
device_type_map = {'Appliance': 0, 'Consumer Electronics': 1, 'IT Equipment': 2}
df['DeviceType_encoded'] = df['Device Type'].map(device_type_map)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

features = ['Condition_encoded', 'DeviceType_encoded',
            'Year of Manufacture', 'Device Age']

metals = ['Gold (g)', 'Silver (g)', 'Aluminum (g)',
          'Platinum (g)', 'Nickel (g)', 'Tin (g)',
          'Lithium (g)', 'Rhodium (g)']

X = df[features]
y = df[metals]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Mean Absolute Error per metal:")
for i, metal in enumerate(metals):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"  {metal}: {mae:.4f}g")


print("Average metal content in dataset:")
for metal in metals:
    print(f"  {metal}: {df[metal].mean():.4f}g")

features_v2 = ['Condition_encoded', 'DeviceType_encoded',
                'Year of Manufacture', 'Device Age',
                'Market Value of Metals']

X2 = df[features_v2]