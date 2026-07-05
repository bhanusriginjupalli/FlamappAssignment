import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\bhanu\Downloads\flamapp\xy_data.csv")

print("Number of points:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

plt.figure(figsize=(8, 8))
plt.scatter(df["x"], df["y"], s=8, color="blue")

plt.title("Scatter Plot of Given XY Points")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")      
plt.grid(True)

plt.show()