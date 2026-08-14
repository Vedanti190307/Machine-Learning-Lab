import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("diamonds.csv")
plt.scatter(df["color"],df["price"])
plt.title("color vs price")
plt.xlabel("color")
plt.ylabel("price")
plt.show()
