import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("C:\\Users\\hana-\\OneDrive\\Desktop\\Task 1\\StudentPerformanceFactors.csv")


print(data.head())
df=data[['Hours_Studied' , 'Exam_Score']]
print(df.isnull().sum())
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='blue')
plt.title('Study Hours vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.savefig("firstplot.png")  
print("Plot saved")

X = df[['Hours_Studied']]  
y = df['Exam_Score']       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()

plt.savefig("plot.png")  
print("Plot saved")


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared (Accuracy):", r2)

Hours_Studied = [[10]]
predicted_score = model.predict(Hours_Studied)
print("Predicted exam score for 10 hours of study:", predicted_score[0])

