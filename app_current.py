import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="IoT Manufacturing Anomaly Detection",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate synthetic IoT data
def generate_sample_data():
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-11-27', periods=1000, freq='T')
    device_ids = ['A1', 'A2', 'A3']
    
    # More varied and realistic data generation
    def generate_variable_with_anomalies(base_mean, base_std, anomaly_periods=50):
        normal_data = np.random.normal(base_mean, base_std, 1000 - anomaly_periods)
        anomaly_data = np.random.normal(base_mean * 1.5, base_std * 2, anomaly_periods)
        return np.concatenate([normal_data, anomaly_data])
    
    data = {
        'timestamp': timestamps,
        'device_id': [np.random.choice(device_ids) for _ in range(1000)],
        'voltage': generate_variable_with_anomalies(220, 2),
        'current': generate_variable_with_anomalies(5, 0.5),
        'temperature': generate_variable_with_anomalies(30, 2),
        'speed': np.random.choice([1400, 1500, 1600], 1000, p=[0.3, 0.5, 0.2]),
        'vibration': np.random.normal(10, 2, 1000),
        'pressure': generate_variable_with_anomalies(100, 5)
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
    data['anomaly'] = data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return data

def filter_by_interval(filtered_data, interval):
    latest_time = filtered_data['timestamp'].max()
    interval_map = {
        "Last Hour": timedelta(hours=1),
        "Last 6 Hours": timedelta(hours=6),
        "Last 12 Hours": timedelta(hours=12),
        "Last 24 Hours": timedelta(hours=24)
    }
    return filtered_data[filtered_data['timestamp'] >= (latest_time - interval_map[interval])] if interval != "Full Dataset" else filtered_data

def create_comprehensive_charts(filtered_data, selected_variables):
    # Prepare color scheme
    color_map = {'Normal': 'blue', 'Anomaly': 'red'}
    
    # Comprehensive Visualization Section
    st.header("Comprehensive IoT Anomaly Visualization")
    
    # Multi-Column Layout for Better Organization
    cols = st.columns(2)
    
    # 1. Step Line Chart
    with cols[0]:
        st.subheader("Step Line Chart")
        for var in selected_variables:
            fig_step = px.line(filtered_data, x='timestamp', y=var, 
                               color='anomaly', line_shape='hv',
                               color_discrete_map=color_map)
            st.plotly_chart(fig_step)
    
    # 2. Range Area Chart
    with cols[1]:
        st.subheader("Range Area Chart")
        for var in selected_variables:
            fig_range = go.Figure()
            fig_range.add_trace(go.Scatter(
                x=filtered_data['timestamp'], 
                y=filtered_data[var] + filtered_data[var].std(), 
                mode='lines', name='Upper Bound', line=dict(color='rgba(255,0,0,0.3)')
            ))
            fig_range.add_trace(go.Scatter(
                x=filtered_data['timestamp'], 
                y=filtered_data[var] - filtered_data[var].std(), 
                mode='lines', fill='tonexty', name='Lower Bound', line=dict(color='rgba(0,0,255,0.3)')
            ))
            fig_range.update_layout(title=f"Range Area for {var.capitalize()}")
            st.plotly_chart(fig_range)
    
    # 3. Brushable Timeline
    st.subheader("Brushable Timeline")
    fig_brush = px.scatter(filtered_data, x='timestamp', y=selected_variables[0], 
                           color='anomaly', color_discrete_map=color_map)
    st.plotly_chart(fig_brush)
    
    # 4. Threshold Chart
    st.subheader("Threshold Chart")
    for var in selected_variables:
        threshold = filtered_data[var].mean() + filtered_data[var].std()
        fig_threshold = go.Figure()
        fig_threshold.add_trace(go.Scatter(
            x=filtered_data['timestamp'], 
            y=filtered_data[var], 
            mode='lines', 
            name=var,
            line=dict(color='blue')
        ))
        fig_threshold.add_hline(y=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig_threshold)
    
    # 5. Radar Chart
    st.subheader("Radar Chart")
    if len(selected_variables) > 1:
        radar_means = filtered_data[selected_variables].mean()
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=radar_means.values,
            theta=radar_means.index,
            fill='toself'
        ))
        st.plotly_chart(fig_radar)
    
    # 6. Radial Bar Chart
    st.subheader("Radial Bar Chart")
    fig_radial = go.Figure(go.Barpolar(
        r=filtered_data[selected_variables[0]],
        theta=filtered_data['timestamp'].dt.hour + filtered_data['timestamp'].dt.minute/60,
        marker_color=filtered_data['anomaly'].map(color_map)
    ))
    st.plotly_chart(fig_radial)
    
    # 7. Heatmap Calendar
    st.subheader("Heatmap Calendar")
    filtered_data['day'] = filtered_data['timestamp'].dt.date
    heatmap_data = filtered_data.groupby('day')[selected_variables[0]].mean().reset_index()
    fig_heatmap = px.density_heatmap(heatmap_data, x='day', y=selected_variables[0])
    st.plotly_chart(fig_heatmap)
    
    # 8. Sparklines
    st.subheader("Sparklines")
    sparkline_df = filtered_data.groupby('device_id')[selected_variables].mean()
    fig_sparkline = go.Figure(data=[
        go.Scatter(x=list(range(len(row))), y=row.values, mode='lines+markers', 
                   name=index) for index, row in sparkline_df.iterrows()
    ])
    st.plotly_chart(fig_sparkline)
    
    # 9. Waterfall Chart
    st.subheader("Waterfall Chart")
    filtered_data['value_diff'] = filtered_data[selected_variables[0]].diff().fillna(0)
    fig_waterfall = go.Figure(go.Waterfall(
        x=filtered_data['timestamp'].dt.strftime('%H:%M:%S').iloc[:20],
        y=filtered_data['value_diff'].iloc[:20],
        increasing={"marker":{"color":"red"}},
        decreasing={"marker":{"color":"green"}}
    ))
    st.plotly_chart(fig_waterfall)
    
    # 10. Box Plot Timeline
    st.subheader("Box Plot Timeline")
    fig_boxplot = go.Figure()
    for var in selected_variables:
        fig_boxplot.add_trace(go.Box(y=filtered_data[var], name=var))
    st.plotly_chart(fig_boxplot)
    
    # 11. Stream Graph
    st.subheader("Stream Graph")
    fig_stream = px.area(
        filtered_data, 
        x='timestamp', 
        y=selected_variables[0],  # Use the first selected variable
        color='device_id', 
        title="Stream Graph"
    )
    st.plotly_chart(fig_stream)

def main():
    # Load or Generate Data
    try:
        data = load_data('iot_data.csv')
    except FileNotFoundError:
        data = generate_sample_data()
    
    # Sidebar Configuration
    st.sidebar.title("IoT Anomaly Detection Dashboard")
    
    # Device Selection
    devices = data['device_id'].unique()
    selected_device = st.sidebar.selectbox("Select Device", devices)
    
    # Time Range Selection with More Options
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last Hour", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Full Dataset"]
    )
    
    # Variables Selection with Multi-Select
    variables = ['voltage', 'current', 'temperature', 'speed', 'vibration', 'pressure']
    selected_variables = st.sidebar.multiselect(
        "Select Variables to Analyze",
        options=variables,
        default=['voltage', 'temperature']
    )
    
    # Additional Interval Selection
    interval_type = st.sidebar.selectbox(
        "Select Interval Type",
        ["Minute-wise", "Hour-wise", "Day-wise"]
    )
    
    # Data Filtering
    filtered_data = data[data['device_id'] == selected_device]
    filtered_data = filter_by_interval(filtered_data, time_range)
    
    if filtered_data.empty:
        st.warning("⚠️ No data available for the selected device and time range.")
        return
    
    # Anomaly Detection
    model = train_anomaly_model(filtered_data, selected_variables)
    if model is None:
        st.error("🚨 Anomaly detection model could not be trained.")
        return
    
    filtered_data = predict_anomalies(model, filtered_data, selected_variables)
    
    # Anomaly Reporting
    anomalies = filtered_data[filtered_data['anomaly'] == 'Anomaly']
    if not anomalies.empty:
        st.error(f"🚨 Anomalies Detected: {len(anomalies)} in the Selected Time Range")
        st.dataframe(anomalies)
    else:
        st.success("✅ No anomalies detected in the selected time range.")
    
    # Comprehensive Visualization
    create_comprehensive_charts(filtered_data, selected_variables)

if __name__ == "__main__":
    main()
