import matplotlib.pyplot as plt
Student=["x","y","z"]
Marks=[10,20,80]
bars=plt.bar(Student,Marks,color=['red','green','purple'])
plt.title("Student Marks")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.show()
