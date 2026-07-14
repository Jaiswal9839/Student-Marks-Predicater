# Import Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Import Dataset
df = pd.read_csv("C:/ds-ml/student_info.csv")
df.head()
 # Data Visualization (Scatter Plot)
x=df.study_hours
y=df.student_marks
plt.scatter(x,y)
plt.title("Study Hours Vs. Student Marks")
plt.xlabel("Study Hours")
plt.ylabel("Student Marks")
plt.show()
# Checking missing value
df.isnull().sum()
 # Handle missing values
df=df.fillna(df.mean())
df.isnull().sum()
 # Separate dataset into independent and dependent variables
X=df.drop("student_marks",axis=1)
X.head()
y=df.drop("study_hours",axis=1)
y.head()
 # Train Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
 # Apply LinearRegression Algorithm
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)
 # Test the model
sh=float(input("Enter study hours : "))
if sh>=4 and sh<=12:
    res=model.predict([[sh]])[0][0].round(2)
    print("Expected Marks=",res)
else:
    print("Invalid Study Hours")