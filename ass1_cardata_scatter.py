import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Car data.csv")
plt.scatter(df["CC"],df["Onroad Price"])
plt.title("CC vs Onroad Price")
plt.xlabel("CC")
plt.ylabel("Onroad Price")
plt.show()
