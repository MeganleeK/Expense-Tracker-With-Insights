import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Expense Tracker Dashboard", layout="wide")

st.title("Expense Tracker & Spending Insights")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank_statement.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df


df = load_data()
expenses = df[df["Amount"] < 0].copy()
expenses["Amount"] = expenses["Amount"].abs()

# Sidebar filters
st.sidebar.header("Filters")
months = st.sidebar.multiselect(
    "Select month(s)",
    options=sorted(expenses["Month"].unique()),
    default=sorted(expenses["Month"].unique())
)

filtered = expenses[expenses["Month"].isin(months)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Spending", f"R {filtered['Amount'].sum():,.2f}")
col2.metric("Transactions", len(filtered))
col3.metric("Average Spend", f"R {filtered['Amount'].mean():,.2f}")

# Category spending
category_spending = filtered.groupby("Category")["Amount"].sum().sort_values(ascending=False)

st.subheader("Spending by Category")
fig1, ax1 = plt.subplots()
category_spending.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Amount")
ax1.set_xlabel("Category")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Monthly trend
monthly_spending = filtered.groupby("Month")["Amount"].sum()

st.subheader("Monthly Spending Trend")
fig2, ax2 = plt.subplots()
monthly_spending.plot(marker="o", ax=ax2)
ax2.set_ylabel("Amount")
ax2.set_xlabel("Month")
st.pyplot(fig2)

# Budget alerts
st.subheader("Budget Alerts")
budgets = {
    "Food": 1000,
    "Shopping": 2000,
    "Transport": 1500,
    "Entertainment": 500
}

for category, limit in budgets.items():
    if category in category_spending.index and category_spending[category] > limit:
        st.warning(f"{category} budget exceeded (R {category_spending[category]:,.2f} / R {limit:,.2f})")

# Prediction
if len(monthly_spending) >= 2:
    X = np.array(range(len(monthly_spending))).reshape(-1, 1)
    y = monthly_spending.values

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[len(monthly_spending)]])
    prediction = model.predict(next_month)[0]

    st.subheader("Next Month Prediction")
    st.metric("Predicted Spending", f"R {prediction:,.2f}")

# Raw data
with st.expander("View raw transactions"):
    st.dataframe(filtered)
