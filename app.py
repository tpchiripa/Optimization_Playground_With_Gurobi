import streamlit as st
from gurobipy import Model, GRB
import pandas as pd
import altair as alt

# -----------------------------
# Page Config + Links
# -----------------------------
st.set_page_config(page_title="Optimization Playground", page_icon="⚡")

st.markdown("""
# ⚡ Optimization Playground with Gurobi  

Built with [Streamlit](https://streamlit.io) + [Gurobi](https://www.gurobi.com/).  

📂 Source Code: [GitHub Repo](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME)  
🌐 Live App: [Click here to try in real-time](https://YOUR-STREAMLIT-URL.streamlit.app)  
---
""")

# -----------------------------
# Default Dataset: Diet Problem
# -----------------------------
default_diet = pd.DataFrame({
    "Food": ["chicken", "beef", "tofu", "rice", "milk", "eggs"],
    "Cost": [2, 3, 1.5, 0.5, 1, 0.8],
    "Calories": [200, 250, 150, 180, 120, 70],
    "Protein": [20, 25, 10, 4, 6, 6],
    "Fat": [5, 15, 8, 1, 5, 5],
})

scenarios = {
    "Balanced (default)": {"cal": 2000, "protein": 50, "fat": 70},
    "Athlete": {"cal": 3000, "protein": 150, "fat": 90},
    "Weight loss": {"cal": 1500, "protein": 80, "fat": 50},
    "Low-carb": {"cal": 2000, "protein": 120, "fat": 100},
    "Budget": {"cal": 1800, "protein": 60, "fat": 60},
}

# -----------------------------
# Default Dataset: Transportation Problem
# -----------------------------
default_transport = pd.DataFrame({
    "Warehouse": ["W1", "W1", "W1", "W2", "W2", "W2"],
    "Store": ["S1", "S2", "S3", "S1", "S2", "S3"],
    "Cost": [2, 4, 5, 3, 2, 6],
    "Supply": [100, None, None, 80, None, None],
    "Demand": [50, 60, 70, None, None, None],
})

# -----------------------------
# Streamlit App
# -----------------------------
problem_choice = st.sidebar.selectbox(
    "Choose Problem",
    ["Diet Problem", "Transportation Problem"]
)

# -----------------------------
# Diet Problem
# -----------------------------
if problem_choice == "Diet Problem":
    st.header("🥗 Diet Optimization")

    uploaded_file = st.file_uploader("Upload Diet CSV (Food, Cost, Calories, Protein, Fat)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            diet_df = pd.read_csv(uploaded_file)
        else:
            diet_df = pd.read_excel(uploaded_file)
    else:
        diet_df = default_diet.copy()

    st.write("### Current Dataset")
    st.dataframe(diet_df)

    scenario_choice = st.sidebar.selectbox("Choose Scenario", list(scenarios.keys()))
    preset = scenarios[scenario_choice]

    st.sidebar.header("Nutritional Requirements")
    min_cal = st.sidebar.number_input("Minimum Calories", 100, 5000, preset["cal"], 50)
    min_protein = st.sidebar.number_input("Minimum Protein (g)", 1, 500, preset["protein"], 1)
    min_fat = st.sidebar.number_input("Minimum Fat (g)", 1, 200, preset["fat"], 1)

    integer_portions = st.sidebar.checkbox("Require whole portions?", value=False)

    if st.button("Optimize Diet"):
        foods = diet_df["Food"].tolist()
        costs = dict(zip(foods, diet_df["Cost"]))
        calories = dict(zip(foods, diet_df["Calories"]))
        protein = dict(zip(foods, diet_df["Protein"]))
        fat = dict(zip(foods, diet_df["Fat"]))

        m = Model("diet")
        m.setParam("OutputFlag", 0)

        vtype = GRB.INTEGER if integer_portions else GRB.CONTINUOUS
        x = m.addVars(foods, name="x", lb=0, vtype=vtype)

        m.setObjective(sum(costs[f] * x[f] for f in foods), GRB.MINIMIZE)

        m.addConstr(sum(calories[f] * x[f] for f in foods) >= min_cal, "calories")
        m.addConstr(sum(protein[f] * x[f] for f in foods) >= min_protein, "protein")
        m.addConstr(sum(fat[f] * x[f] for f in foods) >= min_fat, "fat")

        m.optimize()

        if m.status == GRB.OPTIMAL:
            st.subheader(f"✅ Optimal Diet for {scenario_choice}")

            results = {f: x[f].x for f in foods if x[f].x > 1e-6}
            df = pd.DataFrame(list(results.items()), columns=["Food", "Portions"])

            st.write("### Diet Plan")
            st.dataframe(df, use_container_width=True)

            st.write("### Portions per Food")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Food", sort="-y"),
                y="Portions",
                tooltip=["Food", "Portions"]
            )
            st.altair_chart(chart, use_container_width=True)

            st.success(f"💰 Total cost: {m.objVal:.2f}")
        else:
            st.error("⚠️ No feasible solution found.")

# -----------------------------
# Transportation Problem
# -----------------------------
elif problem_choice == "Transportation Problem":
    st.header("🚚 Transportation Optimization")

    uploaded_file = st.file_uploader("Upload Transportation CSV (Warehouse, Store, Cost, Supply, Demand)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            transport_df = pd.read_csv(uploaded_file)
        else:
            transport_df = pd.read_excel(uploaded_file)
    else:
        transport_df = default_transport.copy()

    st.write("### Current Dataset")
    st.dataframe(transport_df)

    if st.button("Optimize Shipping"):
        warehouses = transport_df["Warehouse"].unique().tolist()
        stores = transport_df["Store"].unique().tolist()

        supply = transport_df.groupby("Warehouse")["Supply"].max().dropna().to_dict()
        demand = transport_df.groupby("Store")["Demand"].max().dropna().to_dict()

        shipping_costs = {(row["Warehouse"], row["Store"]): row["Cost"]
                          for _, row in transport_df.iterrows()}

        m = Model("transport")
        m.setParam("OutputFlag", 0)

        x = m.addVars(warehouses, stores, name="ship", lb=0)

        m.setObjective(
            sum(shipping_costs[w, s] * x[w, s] for w in warehouses for s in stores if (w, s) in shipping_costs),
            GRB.MINIMIZE
        )

        for w in warehouses:
            if w in supply:
                m.addConstr(sum(x[w, s] for s in stores if (w, s) in shipping_costs) <= supply[w], f"supply_{w}")

        for s in stores:
            if s in demand:
                m.addConstr(sum(x[w, s] for w in warehouses if (w, s) in shipping_costs) >= demand[s], f"demand_{s}")

        m.optimize()

        if m.status == GRB.OPTIMAL:
            st.subheader("✅ Optimal Shipping Plan")

            results = []
            for w in warehouses:
                for s in stores:
                    if (w, s) in shipping_costs and x[w, s].x > 1e-6:
                        results.append([w, s, x[w, s].x, shipping_costs[w, s]])

            df = pd.DataFrame(results, columns=["Warehouse", "Store", "Units Shipped", "Cost per Unit"])
            df["Total Cost"] = df["Units Shipped"] * df["Cost per Unit"]

            st.write("### Shipping Plan")
            st.dataframe(df, use_container_width=True)

            st.write("### Shipments per Route")
            chart = alt.Chart(df).mark_bar().encode(
                x="Warehouse",
                y="Units Shipped",
                color="Store",
                tooltip=["Warehouse", "Store", "Units Shipped"]
            )
            st.altair_chart(chart, use_container_width=True)

            st.success(f"💰 Total shipping cost: {m.objVal:.2f}")
        else:
            st.error("⚠️ No feasible solution found.")
