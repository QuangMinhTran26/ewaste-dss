import pandas as pd
df = pd.read_csv("updated_e_waste_dataset.csv")

# Encode Device Condition
condition_map = {'Broken': 0, 'Average': 1, 'Good': 2}
df['Condition_encoded'] = df['Device Condition'].map(condition_map)

# Encode Device Type
device_type_map = {'Appliance': 0, 'Consumer Electronics': 1, 'IT Equipment': 2}
df['DeviceType_encoded'] = df['Device Type'].map(device_type_map)
