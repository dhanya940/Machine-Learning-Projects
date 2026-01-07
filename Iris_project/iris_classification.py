
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("Iris.csv")

print("First 5 rows of dataset:")
print(df.head())


X = df.drop(columns=["Id", "Species"], errors="ignore")
y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


print("\nEnter flower measurements:")

sepal_length = float(input("Sepal Length (cm): "))
sepal_width  = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width  = float(input("Petal Width (cm): "))

sample_flower = [[sepal_length, sepal_width, petal_length, petal_width]]


prediction = model.predict(sample_flower)

print("\nNew Flower Prediction:")
print("Measurements:", sample_flower)
print("Predicted Species:", prediction[0])
