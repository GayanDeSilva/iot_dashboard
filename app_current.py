import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta
from sklearn.ensemble import IsolationForest

# Page configuration
st.set_page_config(
    page_title="IoT Manufacturing Dashboard",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate synthetic IoT data
def generate_sample_data():
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-11-27', periods=1000, freq='T')
    device_ids = ['A1', 'A2', 'A3']
    data = {
        'timestamp': timestamps,
        'device_id': [np.random.choice(device_ids) for _ in range(1000)],
        'voltage': np.concatenate([
            np.random.normal(220, 2, 900), np.random.normal(230, 10, 50), np.random.normal(215, 5, 50)
        ]),
        'current': np.concatenate([
            np.random.normal(5, 0.5, 900), np.random.normal(7, 1, 50), np.random.normal(3, 0.8, 50)
        ]),
        'temperature': np.concatenate([
            np.random.normal(30, 2, 900), np.random.normal(40, 5, 50), np.random.normal(25, 3, 50)
        ]),
        'speed': np.random.choice([1400, 1500, 1600], 1000, p=[0.3, 0.5, 0.2]),
        'vibration': np.random.normal(10, 2, 1000),
        'dependent_var': np.random.normal(7, 0.3, 1000)
    }
    df = pd.DataFrame(data)
    df.to_csv('iot_data.csv', index=False)
    return df

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['timestamp'])

def train_anomaly_model(data, features):
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        st.warning(f"⚠️ Missing features in data: {', '.join(missing_features)}")
        return None
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data[features])
    return model

def predict_anomalies(model, data, features):
    data['anomaly'] = model.predict(data[features])
    data['anomaly'] = data['anomaly'].apply(lambda x: 'Normal' if x == 1 else 'Anomaly')
    return data

def filter_by_interval(filtered_data, interval):
    latest_time = filtered_data['timestamp'].max()
    if interval == "Last Hour":
        return filtered_data[filtered_data['timestamp'] >= (latest_time - timedelta(hours=1))]
    elif interval == "Last 6 Hours":
        return filtered_data[filtered_data['timestamp'] >= (latest_time - timedelta(hours=6))]
    elif interval == "Last 12 Hours":
        return filtered_data[filtered_data['timestamp'] >= (latest_time - timedelta(hours=12))]
    else:
        return filtered_data

def filter_by_variables(filtered_data, selected_variables):
    available_columns = set(filtered_data.columns)
    selected_variables = [var for var in selected_variables if var in available_columns]
    if not selected_variables:
        st.warning("⚠️ No valid variables selected for analysis.")
    return filtered_data[selected_variables + ['timestamp']]

def create_charts(filtered_data, anomalies, selected_variables):
    # Line charts for selected variables
    for variable in selected_variables:
        st.subheader(f"{variable.capitalize()} Over Time with Anomalies")
        fig = px.line(
            filtered_data, x='timestamp', y=variable, color='anomaly', title=f"{variable.capitalize()} Over Time"
        )
        st.plotly_chart(fig)

    st.subheader("Range Area Chart")
    # Range Area Chart
    for variable in selected_variables:
        filtered_data[f'{variable}_high'] = filtered_data[variable] + 5
        filtered_data[f'{variable}_low'] = filtered_data[variable] - 5
        fig_range = go.Figure()
        fig_range.add_trace(go.Scatter(
            x=filtered_data['timestamp'], y=filtered_data[f'{variable}_high'], mode='lines', name='High'
        ))
        fig_range.add_trace(go.Scatter(
            x=filtered_data['timestamp'], y=filtered_data[f'{variable}_low'], mode='lines', fill='tonexty', name='Low'
        ))
        fig_range.update_layout(title=f"Range Area for {variable.capitalize()}")
        st.plotly_chart(fig_range)

    st.subheader("Waterfall Chart")
    # Waterfall Chart
    variable = selected_variables[0] if selected_variables else 'temperature'
    filtered_data['value_diff'] = filtered_data[variable].diff().fillna(0)
    fig_waterfall = go.Figure(go.Waterfall(
        x=filtered_data['timestamp'].dt.strftime('%H:%M:%S').iloc[:20],
        y=filtered_data['value_diff'].iloc[:20],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "red"}},
        decreasing={"marker": {"color": "green"}}
    ))
    fig_waterfall.update_layout(title=f"Waterfall of {variable.capitalize()} Changes")
    st.plotly_chart(fig_waterfall)

    st.subheader("Radar Chart")
    # Radar Chart
    if len(selected_variables) > 1:
        radar_data = filtered_data[selected_variables].mean().reset_index()
        radar_data.columns = ['Metric', 'Value']
        fig_radar = px.line_polar(radar_data, r='Value', theta='Metric', line_close=True, title="Variable Metrics")
        st.plotly_chart(fig_radar)
    else:
        st.warning("⚠️ Please select at least 2 variables to view the Radar Chart.")

    st.subheader("Heatmap Calendar")
    filtered_data['day'] = filtered_data['timestamp'].dt.date
    heatmap_data = filtered_data.groupby('day')[selected_variables[0]].mean().reset_index()
    fig_heatmap = px.density_heatmap(
        heatmap_data, x='day', y=selected_variables[0], title="Daily Heatmap"
    )
    st.plotly_chart(fig_heatmap)

    st.subheader("Brushable Timeline")
    fig_brush = px.scatter(filtered_data, x='timestamp', y=selected_variables[0], color='anomaly', title="Brushable Timeline")
    st.plotly_chart(fig_brush)

    st.subheader("Radial Bar Chart")
    fig_radial = px.bar_polar(filtered_data, r=selected_variables[0], theta=filtered_data['timestamp'].dt.minute, title="Radial Bar Chart")
    st.plotly_chart(fig_radial)

    st.subheader("Stream Graph")
    fig_stream = px.area(filtered_data, x='timestamp', y=selected_variables[0], title="Stream Graph")
    st.plotly_chart(fig_stream)

def main():
    try:
        data = load_data('iot_data.csv')
    except FileNotFoundError:
        data = generate_sample_data()

    st.sidebar.title("IoT Analytics Dashboard")
    devices = data['device_id'].unique()
    selected_device = st.sidebar.selectbox("Select Device", devices)
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last Hour", "Last 6 Hours", "Last 12 Hours", "Full Dataset"]
    )
    variables = ['voltage', 'current', 'temperature', 'vibration']
    selected_variables = st.sidebar.multiselect(
        "Select Variables to Analyze",
        options=variables,
        default=['voltage']
    )

    filtered_data = data[data['device_id'] == selected_device]
    filtered_data = filter_by_interval(filtered_data, time_range)
    filtered_data = filter_by_variables(filtered_data, selected_variables)

    if filtered_data.empty:
        st.warning("⚠️ No data available for the selected device, time range, or variables.")
        return

    model = train_anomaly_model(filtered_data, selected_variables)
    if model is None:
        st.error("🚨 Anomaly detection model could not be trained due to missing features.")
        return

    filtered_data = predict_anomalies(model, filtered_data, selected_variables)
    anomalies = filtered_data[filtered_data['anomaly'] == 'Anomaly']
    if not anomalies.empty:
        st.error(f"🚨 Anomalies Detected: {len(anomalies)} in the Selected Time Range")
        st.dataframe(anomalies) 
    else:
        st.success("✅ No anomalies detected in the selected time range.")
        
    create_charts(filtered_data, anomalies, selected_variables)
    
if __name__ == "__main__":
    main()
    
