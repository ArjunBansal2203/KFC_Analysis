import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Global Fast-Food Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# TITLE & INTRO
# =====================================================
st.title("🍔 Global Fast-Food Sales Analytics Dashboard")
st.markdown(
    "An interactive business intelligence dashboard exploring **sales performance, "
    "customer behavior, marketing efficiency, and growth patterns** across countries (2018–2024)."
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("0210ce92-a19a-457d-8afc-faab602ec0fb.csv")
    return df

df = load_data()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🎛️ Filter Controls")

country_filter = st.sidebar.multiselect(
    "Select Countries",
    options=sorted(df["Country"].unique()),
    default=sorted(df["Country"].unique())
)

year_filter = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2018, 2024)
)

filtered_df = df[
    (df["Country"].isin(country_filter)) &
    (df["Year"].between(year_filter[0], year_filter[1]))
]

# =====================================================
# KPI SECTION
# =====================================================
st.markdown("## 📌 Business Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("💰 Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
k2.metric("👥 Total Customers", f"{filtered_df['Customers'].sum():,.0f}")
k3.metric("📢 Avg Marketing Spend", f"${filtered_df['Marketing_Spend'].mean():,.0f}")
k4.metric("🏪 Active Branches", filtered_df["Branch_ID"].nunique())

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Sales & Growth",
    "👥 Customers & Marketing",
    "📊 Performance Insights"
])

# =====================================================
# TAB 1 — SALES & GROWTH
# =====================================================
with tab1:
    st.subheader("Global Sales Trend Over Time")

    yearly_sales = filtered_df.groupby("Year")["Sales"].sum().reset_index()

    fig1 = px.line(
        yearly_sales,
        x="Year",
        y="Sales",
        markers=True,
        title="Total Sales by Year",
        template="plotly_white"
    )

    fig1.add_annotation(
        text="Sales stagnation followed by strong rebound in recent years",
        xref="paper", yref="paper",
        x=0, y=1.1, showarrow=False
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Country-wise Sales Comparison")

    country_sales = (
        filtered_df.groupby("Country")["Sales"]
        .sum().reset_index().sort_values("Sales")
    )

    fig2 = px.bar(
        country_sales,
        x="Sales",
        y="Country",
        orientation="h",
        title="Total Sales by Country",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# TAB 2 — CUSTOMERS & MARKETING
# =====================================================
with tab2:
    st.subheader("Customer Volume vs Sales")

    fig3 = px.scatter(
        filtered_df,
        x="Customers",
        y="Sales",
        color="Country",
        trendline="ols",
        opacity=0.7,
        title="Relationship Between Customers and Sales",
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Marketing Spend Efficiency")

    filtered_df["Sales_per_Customer"] = (
        filtered_df["Sales"] / filtered_df["Customers"]
    )

    fig4 = px.scatter(
        filtered_df,
        x="Marketing_Spend",
        y="Sales_per_Customer",
        color="Country",
        opacity=0.6,
        title="Marketing Spend vs Sales per Customer",
        template="plotly_white"
    )

    st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# TAB 3 — PERFORMANCE INSIGHTS
# =====================================================
with tab3:
    st.subheader("Revenue Concentration Across Branches")

    branch_sales = (
        filtered_df.groupby("Branch_ID")["Sales"]
        .sum().sort_values(ascending=False)
        .head(15).reset_index()
    )

    fig5 = px.bar(
        branch_sales,
        x="Sales",
        y="Branch_ID",
        orientation="h",
        title="Top 15 Revenue-Generating Branches",
        template="plotly_white"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Sales Distribution")

    fig6 = px.histogram(
        filtered_df,
        x="Sales",
        nbins=40,
        title="Distribution of Sales Values",
        template="plotly_white"
    )

    st.plotly_chart(fig6, use_container_width=True)

# =====================================================
# SIMPLE PREDICTION SECTION
# =====================================================
st.markdown("## 🔮 Sales Prediction Tool")

X = df[["Customers", "Marketing_Spend"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

cust_input = st.slider("Expected Number of Customers", 1500, 6000, 3000)
marketing_input = st.slider("Marketing Spend ($)", 5000, 30000, 15000)

predicted_sales = model.predict([[cust_input, marketing_input]])

st.success(f"📈 Predicted Sales: ${predicted_sales[0]:,.2f}")

# =====================================================
# FINAL INSIGHTS
# =====================================================
st.markdown("""
## 🧠 Key Takeaways

• Customer volume is the **strongest driver** of sales  
• Marketing improves both **revenue and efficiency**  
• Sales growth varies significantly by country  
• A small number of branches generate a large share of revenue  
• Data-driven insights can support **pricing, expansion, and marketing strategy**
""")
