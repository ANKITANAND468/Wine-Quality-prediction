# src/load_data.py

# pandas is a library for working with tabular data (like Excel, but in code)
import pandas as pd

# Load the red wine dataset.
# The UCI file uses semicolons (;) instead of commas to separate columns,
# so we tell pandas that with sep=";"
red_wine = pd.read_csv("data/raw/winequality-red.csv", sep=";")

# Load the white wine dataset the same way
white_wine = pd.read_csv("data/raw/winequality-white.csv", sep=";")

# .head() shows the first 5 rows so we can eyeball the data
print("RED WINE — first 5 rows:")
print(red_wine.head())

print("\nWhite wine shape (rows, columns):", white_wine.shape)
print("Red wine shape (rows, columns):", red_wine.shape)