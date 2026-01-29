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
st.title("🍔 KFC Global Sales Analytics Dashboard")
st.markdown(
    "An interactive business intelligence dashboard exploring **sales performance, "
    "customer behavior, marketing efficiency, and growth patterns** across countries (2018–2024)."
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("KFC_Past_Sales_Dataset.csv")
    return df

df = load_data()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🎛️ Dashboard Filters")

# Country filter
country_filter = st.sidebar.multiselect(
    "Select Country(s)",
    options=sorted(df["Country"].unique()),
    default=sorted(df["Country"].unique())
)

# Year range filter
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

# Month filter
month_filter = st.sidebar.multiselect(
    "Select Month(s)",
    options=sorted(df["Month"].unique()),
    default=sorted(df["Month"].unique())
)

# Marketing spend filter
marketing_range = st.sidebar.slider(
    "Marketing Spend Range",
    int(df["Marketing_Spend"].min()),
    int(df["Marketing_Spend"].max()),
    (
        int(df["Marketing_Spend"].min()),
        int(df["Marketing_Spend"].max())
    )
)

# Customer volume filter
customer_range = st.sidebar.slider(
    "Customer Count Range",
    int(df["Customers"].min()),
    int(df["Customers"].max()),
    (
        int(df["Customers"].min()),
        int(df["Customers"].max())
    )
)

# Apply all filters
filtered_df = df[
    (df["Country"].isin(country_filter)) &
    (df["Year"].between(year_range[0], year_range[1])) &
    (df["Month"].isin(month_filter)) &
    (df["Marketing_Spend"].between(marketing_range[0], marketing_range[1])) &
    (df["Customers"].between(customer_range[0], customer_range[1]))
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
    
    st.subheader("📊 Sales Distribution")
    
    fig6 = px.histogram(
        filtered_df,
        x="Sales",
        nbins=30,
        histnorm="probability density",
        opacity=0.75,
        title="Sales Distribution Across All Branches",
        template="plotly_white"
    )
    
    fig6.update_traces(
        marker_color="#4C78A8",
        marker_line_width=1,
        marker_line_color="white"
    )
    
    fig6.update_layout(
        xaxis_title="Sales Amount",
        yaxis_title="Density",
        bargap=0.05,
        title_font_size=20
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






