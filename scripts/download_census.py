# scripts/download_census_population.py
import pandas as pd
import requests
import os

os.makedirs("data/census", exist_ok=True)

# 11 = DC FIPS code
url = "https://api.census.gov/data/2022/acs/acs5?get=B01003_001E,NAME&for=tract:*&in=state:11"
r = requests.get(url).json()
cols = r[0]
data = r[1:]
df = pd.DataFrame(data, columns=cols)
df.rename(columns={"B01003_001E": "population"}, inplace=True)
df.to_csv("data/census/dc_population.csv", index=False)
print("✅ Census population saved.")
