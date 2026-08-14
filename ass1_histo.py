import matplotlib.pyplot as plt
Marks=[70,10,50,60,30,80,20]
plt.hist(Marks,bins=5,edgecolor='pink')
plt.title("Histogram of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()
