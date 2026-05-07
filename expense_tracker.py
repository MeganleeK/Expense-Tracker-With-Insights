import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv("bank_statement.csv")

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Create Month column
df["Month"] = df["Date"].dt.to_period("M")

# Show first rows
print(df.head())


# FILTER EXPENSES ONLY
expenses = df[df["Amount"] < 0]

# Convert to positive values for easier analysis
expenses["Amount"] = expenses["Amount"].abs()

# TOTAL SPENDING BY CATEGORY
category_spending = expenses.groupby("Category")["Amount"].sum()

print("\nTotal Spending by Category:")
print(category_spending)

# Spending insights
highest_category = expenses.groupby("Category")["Amount"].sum().idxmax()
highest_amount = expenses.groupby("Category")["Amount"].sum().max()
print(f"\nHighest Spending Category: {highest_category} (R{highest_amount:.2f})")

lowest_category = expenses.groupby("Category")["Amount"].sum().idxmin()
lowest_amount = expenses.groupby("Category")["Amount"].sum().min()
print(f"Lowest Spending Category: {lowest_category} (R{lowest_amount:.2f})")

#Budget Alerts
# Budget limits (Dictionary and loop)
budgets = {
    "Food": 1000,
    "Shopping": 2000,
    "Transport": 1500,
    "Entertainment": 500
}

# Check budgets
for category, limit in budgets.items():

    if category_spending[category] > limit:

        print(f"⚠️ {category} budget exceeded!")

# MONTHLY SPENDING TREND
monthly_spending = expenses.groupby("Month")["Amount"].sum()

print("\nMonthly Spending:")
print(monthly_spending)


# VISUALIZATION 1
# Category Spending
category_spending.plot(kind="bar")

plt.title("Total Spending by Category")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# VISUALIZATION 2
# Monthly Trend
monthly_spending.plot(marker="o")

plt.title("Monthly Spending Trend")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.grid(True)
plt.tight_layout()
plt.show()

# VISUALIZATION 3
# Pie Chart of Spending by Category
category_spending.plot(kind="pie", autopct="%1.1f%%")
plt.title('Spending by Category', pad=20)
plt.ylabel("")
plt.show()

# -----------------------------
# PREDICT NEXT MONTH SPENDING
# -----------------------------

# Prepare data
monthly_values = monthly_spending.values

X = np.array(range(len(monthly_values))).reshape(-1, 1)
y = monthly_values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month
next_month = np.array([[len(monthly_values)]])
prediction = model.predict(next_month)

print("\nPredicted Next Month Spending:")
print(round(prediction[0], 2))