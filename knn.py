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


X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42)


knn_v2 = KNeighborsRegressor(n_neighbors=5)
knn_v2.fit(X2_train, y2_train)

y2_pred = knn_v2.predict(X2_test)

# Compare MAE
print("MAE comparison — old vs new model:")
print(f"{'Metal':<15} {'Old MAE':>10} {'New MAE':>10} {'Improvement':>12}")
print("-" * 50)
for i, metal in enumerate(metals):
    old_mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    new_mae = mean_absolute_error(y2_test.iloc[:, i], y2_pred[:, i])
    improvement = ((old_mae - new_mae) / old_mae) * 100
    print(f"{metal:<15} {old_mae:>10.4f} {new_mae:>10.4f} {improvement:>11.1f}%")

    metal_prices = {
    'Gold (g)':     85000 / 1000,  # per gram
    'Silver (g)':   900 / 1000,
    'Aluminum (g)': 2.5 / 1000,
    'Platinum (g)': 32000 / 1000,
    'Nickel (g)':   14 / 1000,
    'Tin (g)':      26 / 1000,
    'Lithium (g)':  13 / 1000,
    'Rhodium (g)':  4700 / 1000,
}

recovery_rate = 0.90  # 90% standard recovery assumption

scenarios = {
    'Pessimistic': 0.80,  # prices 20% below current
    'Base':        1.00,  # current dataset prices
    'Optimistic':  1.20   # prices 20% above current
}

def evaluate_device_scenarios(device_features, cost_of_recovery):
    print("=" * 55)
    print("SCENARIO ANALYSIS")
    print("=" * 55)

    import pandas as pd
    X_new = pd.DataFrame([device_features])
    predicted_metals = knn_v2.predict(X_new)[0]

    results = {}

    for scenario_name, price_multiplier in scenarios.items():
        revenue = 0
        for i, metal in enumerate(metals):
            grams = predicted_metals[i]
            price = metal_prices[metal] * price_multiplier
            value = grams * price * recovery_rate
            revenue += value

        npv = revenue - cost_of_recovery
        results[scenario_name] = npv

        if npv > 0:
            decision = "DISASSEMBLE"
        else:
            decision = "MANUAL CHECK"

        print(f"\n{scenario_name} scenario (prices ×{price_multiplier}):")
        print(f"  NPV:        ${npv:.2f}")
        print(f"  Decision:   {decision}")

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Pessimistic NPV: ${results['Pessimistic']:.2f}")
    print(f"  Base NPV:        ${results['Base']:.2f}")
    print(f"  Optimistic NPV:  ${results['Optimistic']:.2f}")

    # Overall recommendation
    if all(v > 0 for v in results.values()):
        print("\n STRONG RECOMMENDATION: DISASSEMBLE")
        print("   Profitable under all three scenarios")
    elif results['Base'] > 0:
        print("\n CONDITIONAL RECOMMENDATION: DISASSEMBLE")
        print("   Profitable under base and optimistic scenario")
        print("   Monitor price risk carefully")
    else:
        print("\n RECOMMENDATION: MANUAL CHECK")
        print("   Not profitable under base scenario")

sample_device = {
    'Condition_encoded': 1,        # Average
    'DeviceType_encoded': 1,       # Consumer Electronics
    'Year of Manufacture': 2019,
    'Device Age': 5,
    'Market Value of Metals': 300
}

evaluate_device_scenarios(sample_device, cost_of_recovery=25)


iphone_search = df[df['Item Name'].str.contains('iPhone 12', case=False, na=False)]
print(f"Found {len(iphone_search)} rows")
print(iphone_search[['Item Name', 'Brand Name', 'Device Condition', 'Gold (g)']].head())