
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATA

DATA_PATH ="data/car data.csv"
df = pd.read_csv(DATA_PATH)

print("\n📌 Dataset Loaded Successfully")
print("📌 Columns:", df.columns.tolist())
print(df.head())


df["brand"] = df["Car_Name"].astype(str).apply(lambda x: x.split(" ")[0].lower())
df.drop(columns=["Car_Name"], inplace=True)


CURRENT_YEAR = 2025
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df.drop(columns=["Year"], inplace=True)

print("\n✅ Feature Engineering Completed")


TARGET = "Selling_Price"

X = df.drop(TARGET, axis=1)
y = df[TARGET]


num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

print("\n📌 Numerical Features:", list(num_features))
print("📌 Categorical Features:", list(cat_features))


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n📌 Train-Test Split Completed")


model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", LinearRegression())
    ]
)

model.fit(X_train, y_train)
print("\n✅ Model Training Completed")


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Actual Selling Price (Lakhs)")
plt.ylabel("Predicted Selling Price (Lakhs)")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()


sample_car = X_test.iloc[[0]]
predicted_price = model.predict(sample_car)[0]

print("\n🚘 SAMPLE CAR DETAILS")
print(sample_car)
print(f"💰 Predicted Selling Price: ₹ {predicted_price:.2f} Lakhs")


print("""
🌍 REAL-WORLD APPLICATIONS
• Used car resale price prediction
• Online car marketplaces (CarDekho, OLX)
• Insurance & loan valuation
• Dealer pricing intelligence
• Automotive market analytics
""")
