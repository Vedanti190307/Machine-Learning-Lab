import matplotlib.pyplot as plt
Subject=["Python","Java","DS"]
Marks=[10,20,30]
plt.pie(Marks,labels=Subject,autopct="%1.1f%%",startangle=90)
plt.title("Student Marks")
plt.show()
