# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary . 
4. Calculate the y-prediction. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Suvetha k
RegisterNumber: 212225040444 
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.cost_history = []

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.W) + self.b
            y_pred = sigmoid(linear_model)

            cost = -(1/self.m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
            self.cost_history.append(cost)

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
X = np.array([
    [8.5, 120], [6.2, 100], [7.8, 110], [5.5, 85],
    [9.0, 130], [6.0, 95], [7.0, 105], [8.8, 125]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 1])

X = (X - X.mean(axis=0)) / X.std(axis=0)

model = LogisticRegressionGD(lr=0.1, epochs=2000)
model.fit(X, y)

y_pred = model.predict(X)
print("Predictions:", y_pred)
print("Actual     :", y)
print("Accuracy   :", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

plt.plot(model.cost_history)
plt.title("Cost Function Convergence")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Loss)")
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2 = -(model.W[0]*x1 + model.b) / model.W[1]  # boundary equation
plt.plot(x1, x2, color="green", label="Decision Boundary")
plt.legend()
plt.title("Decision Boundary for Logistic Regression (GD)")
plt.xlabel("Feature 1 (scaled CGPA)")
plt.ylabel("Feature 2 (scaled IQ)")
plt.show()
)
```

## Output:
<img width="782" height="571" alt="Screenshot 2026-05-03 201303" src="https://github.com/user-attachments/assets/3e3419fc-4185-4250-96ee-9bb0dddcfd48" />
<img width="820" height="561" alt="Screenshot 2026-05-03 201315" src="https://github.com/user-attachments/assets/4a7e1903-e81a-4c06-a6c0-95fd4668a375" />
<img width="782" height="571" alt="Screenshot 2026-05-03 201303" src="https://github.com/user-attachments/assets/e1d28221-5a3e-43c1-9d71-0fb86834336e" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

