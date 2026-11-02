import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==================================
# 1. PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Analytics & Prediction Dashboard")
st.write("Weather Data Analysis and Rainfall Prediction System")

sns.set_style("whitegrid")

# ==================================
# 2. LOAD DATASET
# ==================================

@st.cache_data
def load_data():
    data = pd.read_csv("weather_data_cleaned.xls")
    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )
    return data

df = load_data()

# Load trained model
model = None

if os.path.exists("best_rainfall_prediction_model.pkl"):
    model = joblib.load(
        "best_rainfall_prediction_model.pkl"
    )

# ==================================
# 3. SIDEBAR NAVIGATION
# ==================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Weather Analysis",
        "District Analysis",
        "Rainfall Prediction"
    ]
)

# ==================================
# 4. OVERVIEW PAGE
# ==================================

if page == "Overview":

    st.header("Weather Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Total Districts",
        df["District"].nunique()
    )
    col4.metric(
        "Average Rainfall",
        f"{df['Rain (mm)'].mean():.2f} mm"
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    st.bar_chart(missing)

# ==================================
# 5. WEATHER ANALYSIS PAGE
# ==================================

elif page == "Weather Analysis":

    st.header("Weather Data Analysis")

    st.subheader("Rainfall Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df["Rain (mm)"].dropna(),
        bins=50,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Monthly Average Rainfall")

    monthly = df.groupby(
        df["Date"].dt.month
    )["Rain (mm)"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rainfall (mm)")
    ax.set_xticks(range(1, 13))
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Temperature Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df["Min Temp (°C)"].dropna(),
        label="Minimum Temperature",
        kde=True,
        ax=ax
    )

    sns.histplot(
        df["Max Temp (°C)"].dropna(),
        label="Maximum Temperature",
        kde=True,
        ax=ax
    )

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    numeric = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

# ==================================
# 6. DISTRICT ANALYSIS PAGE
# ==================================

elif page == "District Analysis":

    st.header("District-wise Weather Analysis")

    districts = sorted(
        df["District"].dropna().unique()
    )

    selected_district = st.selectbox(
        "Select District",
        districts
    )

    district_data = df[
        df["District"] == selected_district
    ]

    st.subheader(
        f"Weather Statistics: {selected_district}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Rainfall",
        f"{district_data['Rain (mm)'].mean():.2f} mm"
    )

    col2.metric(
        "Average Temperature",
        f"{district_data['Max Temp (°C)'].mean():.2f} °C"
    )

    col3.metric(
        "Average Humidity",
        f"{district_data['Max Humidity (%)'].mean():.2f} %"
    )

    st.subheader("District Rainfall Trend")

    monthly_district = district_data.groupby(
        district_data["Date"].dt.month
    )["Rain (mm)"].mean()

    st.line_chart(monthly_district)

    st.subheader("Top 10 Districts by Average Rainfall")

    top_districts = (
        df.groupby("District")["Rain (mm)"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_districts)

# ==================================
# 7. RAINFALL PREDICTION PAGE
# ==================================

elif page == "Rainfall Prediction":

    st.header("🌧️ Rainfall Prediction")

    st.write(
        "Enter weather measurements to estimate rainfall."
    )

    if model is None:

        st.error(
            "Trained model not found. "
            "Please run your machine learning training "
            "script and save the model first."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            min_temp = st.number_input(
                "Minimum Temperature (°C)",
                value=20.0
            )

            max_temp = st.number_input(
                "Maximum Temperature (°C)",
                value=30.0
            )

            min_humidity = st.number_input(
                "Minimum Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0
            )

            max_humidity = st.number_input(
                "Maximum Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

        with col2:

            min_wind = st.number_input(
                "Minimum Wind Speed (Kmph)",
                min_value=0.0,
                value=5.0
            )

            max_wind = st.number_input(
                "Maximum Wind Speed (Kmph)",
                min_value=0.0,
                value=15.0
            )

            selected_date = st.date_input(
                "Select Date"
            )

        if st.button("Predict Rainfall"):

            if min_temp > max_temp:
                st.error("Minimum temperature cannot exceed maximum temperature.")

            elif min_humidity > max_humidity:
                st.error("Minimum humidity cannot exceed maximum humidity.")

            elif min_wind > max_wind:
                st.error("Minimum wind speed cannot exceed maximum wind speed.")

            else:

                month = selected_date.month
                day_of_year = selected_date.timetuple().tm_yday

                input_data = pd.DataFrame([{
                    "Min Temp (°C)": min_temp,
                    "Max Temp (°C)": max_temp,
                    "Min Humidity (%)": min_humidity,
                    "Max Humidity (%)": max_humidity,
                    "Min Wind Speed (Kmph)": min_wind,
                    "Max Wind Speed (Kmph)": max_wind,
                    "Month": month,
                    "Day_of_Year": day_of_year
                }])

                prediction = model.predict(input_data)[0]

                prediction = max(0, prediction)

                st.success("Prediction Completed!")

                st.metric(
                    "Predicted Rainfall",
                    f"{prediction:.2f} mm"
                )

                if prediction == 0:
                    st.info("The model predicts no measurable rainfall.")
                else:
                    st.info(
                        f"Estimated rainfall: {prediction:.2f} mm"
                    )

# ==================================
# 8. FOOTER
# ==================================

st.sidebar.markdown("---")
st.sidebar.write("Weather Prediction Project")
st.sidebar.caption("Developed using Python & Streamlit")
