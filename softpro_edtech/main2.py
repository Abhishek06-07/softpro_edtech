# Full Year Business Forecast (2025).
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

st.title("Full Year Business Forecast (2025)")

# Load The Data.
uploaded_file = st.file_uploader("Upload MIS Data Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
    df['Reg.Date'] = pd.to_datetime(df['Reg.Date'], errors='coerce')
    df['YearMonth'] = df['Reg.Date'].dt.to_period('M')

    # Filter Options.
    technologies = df['Subject'].dropna().unique()
    colleges = df['College'].dropna().unique()
    locations = df['Location'].dropna().unique()

    # Create dropdown.
    selected_tech = st.selectbox("Select Technology", sorted(technologies))
    selected_college = st.selectbox("Select College (Optional)", ['All'] + sorted(colleges.tolist()))
    selected_location = st.selectbox("Select Location (Optional)", ['All'] + sorted(locations.tolist()))

    # Apply filters.
    data = df[df['Subject'] == selected_tech]
    if selected_college != 'All':
        data = data[data['College'] == selected_college]
    if selected_location != 'All':
        data = data[data['Location'] == selected_location]

    # Group by YearMonth and count.
    monthly = data.groupby('YearMonth').size().reset_index(name='SNo.')
    monthly = monthly.set_index('YearMonth').asfreq('M').fillna(0)
    monthly.index = monthly.index.to_timestamp()

    # Create model
    if len(monthly) >= 2:
        # Prepare regression model
        X = np.array([d.toordinal() for d in monthly.index]).reshape(-1, 1)
        y = monthly['SNo.'].values
        model = LinearRegression()
        model.fit(X, y)

        # Now do prediction for all months of 2025
        future_dates = pd.date_range(start="2025-01-01", end="2025-12-01", freq="MS")
        X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)  # FIXED
        y_pred = model.predict(X_future)

        # Show predicted values.
        forcast_df = pd.DataFrame({
            'Month': future_dates.strftime('%B %Y'),
            'Predicted Enrollments': np.round(y_pred).astype(int)
        })
        st.subheader("Monthly Predicted Enrollments for 2025")
        st.dataframe(forcast_df)

        # Show total predictions
        st.success(f"Total predicted enrollments for 2025: {int(np.round(y_pred).sum())}")

    else:
        st.warning("Not enough data for prediction")
