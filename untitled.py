import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Automotive CX Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_uploaded_data(uploaded_files):
    """Load datasets from uploaded files"""
    datasets = {}
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            datasets[uploaded_file.name] = df
    
    return datasets

@st.cache_data
def load_default_data():
    """Load default datasets with caching for better performance"""
    try:
        customers_df = pd.read_csv('automotive_customers.csv')
        services_df = pd.read_csv('automotive_services.csv')
        centers_df = pd.read_csv('service_centers.csv')
        feedback_df = pd.read_csv('customer_feedback.csv')
        
        # Convert date columns
        customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
        services_df['service_date'] = pd.to_datetime(services_df['service_date'])
        feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'])
        
        return customers_df, services_df, centers_df, feedback_df
    except FileNotFoundError:
        return None, None, None, None

def calculate_nps(scores):
    """Calculate Net Promoter Score"""
    if len(scores) == 0:
        return 0
    promoters = len(scores[scores >= 9])
    detractors = len(scores[scores <= 6])
    return ((promoters - detractors) / len(scores)) * 100

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': max_value * 0.8},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Automotive Customer Experience Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.markdown('<div class="sidebar-header">Data Upload</div>', unsafe_allow_html=True)
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload your CSV files (customers, services, centers, feedback)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload the automotive service data CSV files to analyze your company's performance"
    )
    
    # Load data
    customers_df = None
    services_df = None
    centers_df = None
    feedback_df = None
    
    if uploaded_files:
        datasets = load_uploaded_data(uploaded_files)
        
        # Map uploaded files to dataframes
        for filename, df in datasets.items():
            if 'customer' in filename.lower():
                customers_df = df
                if 'registration_date' in df.columns:
                    customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
            elif 'service' in filename.lower() and 'center' not in filename.lower():
                services_df = df
                if 'service_date' in df.columns:
                    services_df['service_date'] = pd.to_datetime(services_df['service_date'])
            elif 'center' in filename.lower():
                centers_df = df
            elif 'feedback' in filename.lower():
                feedback_df = df
                if 'feedback_date' in df.columns:
                    feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'])
        
        st.sidebar.success(f"Loaded {len(datasets)} files successfully!")
        
        # Display dataset info
        for filename, df in datasets.items():
            st.sidebar.write(f"**{filename}**: {len(df)} records")
    
    else:
        # Try to load default data
        customers_df, services_df, centers_df, feedback_df = load_default_data()
        
        if customers_df is None:
            st.warning("Please upload your CSV files to begin analysis, or ensure default files are in the directory.")
            st.info("""
            **Required CSV files:**
            - Customer data (with columns: customer_id, age, city, annual_income, car_brand, etc.)
            - Service data (with columns: service_id, customer_id, service_date, service_type, total_cost, overall_satisfaction, etc.)
            - Service centers data (with columns: service_center_id, center_name, city, etc.)
            - Feedback data (with columns: feedback_id, service_id, complaint_category, resolution_status, etc.)
            """)
            st.stop()
        else:
            st.sidebar.info("Using default dataset files")
    
    # Data validation
    if any(df is None for df in [customers_df, services_df, centers_df, feedback_df]):
        st.error("Some required datasets are missing. Please ensure all CSV files are uploaded.")
        st.stop()
    
    # Sidebar for navigation and filters
    st.sidebar.markdown('<div class="sidebar-header">Navigation & Filters</div>', unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Executive Overview", "Service Analytics", "Customer Insights", 
         "Service Center Performance", "Feedback Analysis", "Predictive Analytics"]
    )
    
    # Filters
    st.sidebar.markdown("### Filters")
    
    # Date range filter
    if 'service_date' in services_df.columns:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[services_df['service_date'].min().date(), services_df['service_date'].max().date()],
            min_value=services_df['service_date'].min().date(),
            max_value=services_df['service_date'].max().date()
        )
    else:
        date_range = None
    
    # City filter
    if 'city' in customers_df.columns:
        selected_cities = st.sidebar.multiselect(
            "Select Cities",
            options=customers_df['city'].unique(),
            default=customers_df['city'].unique()
        )
    else:
        selected_cities = []
    
    # Service type filter
    if 'service_type' in services_df.columns:
        selected_services = st.sidebar.multiselect(
            "Select Service Types",
            options=services_df['service_type'].unique(),
            default=services_df['service_type'].unique()
        )
    else:
        selected_services = []
    
    # Apply filters
    filtered_services = services_df.copy()
    filtered_customers = customers_df.copy()
    
    if date_range and len(date_range) == 2 and 'service_date' in services_df.columns:
        filtered_services = filtered_services[
            (filtered_services['service_date'].dt.date >= date_range[0]) &
            (filtered_services['service_date'].dt.date <= date_range[1])
        ]
    
    if selected_cities and 'city' in filtered_services.columns:
        filtered_services = filtered_services[filtered_services['city'].isin(selected_cities)]
        filtered_customers = filtered_customers[filtered_customers['city'].isin(selected_cities)]
    
    if selected_services and 'service_type' in filtered_services.columns:
        filtered_services = filtered_services[filtered_services['service_type'].isin(selected_services)]
    
    # Dashboard routing
    if page == "Executive Overview":
        executive_overview(filtered_services, filtered_customers, centers_df, feedback_df)
    elif page == "Service Analytics":
        service_analytics(filtered_services, filtered_customers)
    elif page == "Customer Insights":
        customer_insights(filtered_customers, filtered_services)
    elif page == "Service Center Performance":
        service_center_performance(filtered_services, centers_df)
    elif page == "Feedback Analysis":
        feedback_analysis(feedback_df, filtered_services)
    elif page == "Predictive Analytics":
        predictive_analytics(filtered_services, filtered_customers)

def executive_overview(services_df, customers_df, centers_df, feedback_df):
    st.header("Executive Overview Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_customers = len(customers_df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_services = len(services_df)
        st.metric("Total Services", f"{total_services:,}")
    
    with col3:
        if 'overall_satisfaction' in services_df.columns:
            avg_satisfaction = services_df['overall_satisfaction'].mean()
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/5.0")
        else:
            st.metric("Avg Satisfaction", "N/A")
    
    with col4:
        if 'total_cost' in services_df.columns:
            total_revenue = services_df['total_cost'].sum()
            st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
        else:
            st.metric("Total Revenue", "N/A")
    
    with col5:
        if 'recommendation_score' in services_df.columns:
            avg_nps = calculate_nps(services_df['recommendation_score'])
            st.metric("NPS Score", f"{avg_nps:.1f}")
        else:
            st.metric("NPS Score", "N/A")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue Trend
        if 'service_date' in services_df.columns and 'total_cost' in services_df.columns:
            monthly_revenue = services_df.groupby(services_df['service_date'].dt.to_period('M'))['total_cost'].sum()
            if len(monthly_revenue) > 0:
                fig_revenue = px.line(
                    x=monthly_revenue.index.astype(str), 
                    y=monthly_revenue.values,
                    title="Monthly Revenue Trend",
                    labels={'x': 'Month', 'y': 'Revenue (₹)'}
                )
                fig_revenue.update_traces(line_color='#1f77b4', line_width=3)
                st.plotly_chart(fig_revenue, use_container_width=True)
            else:
                st.info("No revenue data available for the selected period")
        else:
            st.info("Revenue trend data not available")
    
    with col2:
        # Service Type Distribution
        if 'service_type' in services_df.columns:
            service_dist = services_df['service_type'].value_counts()
            if len(service_dist) > 0:
                fig_pie = px.pie(
                    values=service_dist.values, 
                    names=service_dist.index,
                    title="Service Type Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No service type data available")
        else:
            st.info("Service type data not available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Satisfaction by City
        if 'city' in services_df.columns and 'overall_satisfaction' in services_df.columns:
            city_satisfaction = services_df.groupby('city')['overall_satisfaction'].mean().sort_values(ascending=True)
            if len(city_satisfaction) > 0:
                fig_city = px.bar(
                    x=city_satisfaction.values,
                    y=city_satisfaction.index,
                    orientation='h',
                    title="Average Satisfaction by City",
                    labels={'x': 'Average Satisfaction', 'y': 'City'}
                )
                st.plotly_chart(fig_city, use_container_width=True)
            else:
                st.info("No city satisfaction data available")
        else:
            st.info("City satisfaction data not available")
    
    with col2:
        # Top Performing Service Centers
        if 'service_center' in services_df.columns and 'overall_satisfaction' in services_df.columns and 'total_cost' in services_df.columns:
            top_centers = services_df.groupby('service_center').agg({
                'overall_satisfaction': 'mean',
                'total_cost': 'sum'
            }).sort_values('overall_satisfaction', ascending=False).head(10)
            
            if len(top_centers) > 0:
                fig_centers = px.scatter(
                    top_centers,
                    x='total_cost',
                    y='overall_satisfaction',
                    title="Service Centers: Revenue vs Satisfaction",
                    labels={'total_cost': 'Total Revenue (₹)', 'overall_satisfaction': 'Avg Satisfaction'},
                    hover_data={'total_cost': ':,.0f'}
                )
                st.plotly_chart(fig_centers, use_container_width=True)
            else:
                st.info("No service center data available")
        else:
            st.info("Service center performance data not available")

def service_analytics(services_df, customers_df):
    st.header("Service Analytics Dashboard")
    
    # Service Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'actual_duration_hours' in services_df.columns:
            avg_duration = services_df['actual_duration_hours'].mean()
            st.metric("Avg Service Duration", f"{avg_duration:.1f} hrs")
        else:
            st.metric("Avg Service Duration", "N/A")
    
    with col2:
        if 'delay_hours' in services_df.columns:
            avg_delay = services_df['delay_hours'].mean()
            st.metric("Avg Delay", f"{avg_delay:.1f} hrs")
        else:
            st.metric("Avg Delay", "N/A")
    
    with col3:
        if 'issue_resolved_first_visit' in services_df.columns:
            first_time_resolution = (services_df['issue_resolved_first_visit'].sum() / len(services_df)) * 100
            st.metric("First-Time Resolution", f"{first_time_resolution:.1f}%")
        else:
            st.metric("First-Time Resolution", "N/A")
    
    with col4:
        if 'total_cost' in services_df.columns:
            avg_cost = services_df['total_cost'].mean()
            st.metric("Avg Service Cost", f"₹{avg_cost:,.0f}")
        else:
            st.metric("Avg Service Cost", "N/A")
    
    # Service Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Service Duration vs Satisfaction
        if 'actual_duration_hours' in services_df.columns and 'overall_satisfaction' in services_df.columns:
            fig_duration = px.scatter(
                services_df.sample(min(1000, len(services_df))),  # Sample for better performance
                x='actual_duration_hours',
                y='overall_satisfaction',
                color='service_type' if 'service_type' in services_df.columns else None,
                title="Service Duration vs Customer Satisfaction",
                labels={'actual_duration_hours': 'Duration (hours)', 'overall_satisfaction': 'Satisfaction'}
            )
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.info("Duration vs satisfaction data not available")
    
    with col2:
        # Cost vs Satisfaction by Service Type
        if 'service_type' in services_df.columns and 'total_cost' in services_df.columns and 'overall_satisfaction' in services_df.columns:
            avg_by_service = services_df.groupby('service_type').agg({
                'total_cost': 'mean',
                'overall_satisfaction': 'mean'
            }).reset_index()
            
            fig_cost_sat = px.scatter(
                avg_by_service,
                x='total_cost',
                y='overall_satisfaction',
                text='service_type',
                title="Average Cost vs Satisfaction by Service Type",
                labels={'total_cost': 'Average Cost (₹)', 'overall_satisfaction': 'Avg Satisfaction'}
            )
            fig_cost_sat.update_traces(textposition="top center")
            st.plotly_chart(fig_cost_sat, use_container_width=True)
        else:
            st.info("Cost vs satisfaction data not available")
    
    # Detailed Service Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Delay Analysis
        if 'delay_hours' in services_df.columns and 'overall_satisfaction' in services_df.columns:
            # Create more varied delay categories with realistic distribution
            services_df_temp = services_df.copy()
            services_df_temp['delay_category'] = pd.cut(
                services_df_temp['delay_hours'], 
                bins=[-0.1, 0, 1, 3, float('inf')], 
                labels=['No Delay', 'Minor (<1h)', 'Moderate (1-3h)', 'Major (>3h)']
            )
            delay_satisfaction = services_df_temp.groupby('delay_category')['overall_satisfaction'].mean()
            
            fig_delay = px.bar(
                x=delay_satisfaction.index,
                y=delay_satisfaction.values,
                title="Impact of Delays on Customer Satisfaction",
                labels={'x': 'Delay Category', 'y': 'Average Satisfaction'},
                color=delay_satisfaction.values,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_delay, use_container_width=True)
        else:
            st.info("Delay analysis data not available")
    
    with col2:
        # Parts Availability Impact
        if 'parts_availability' in services_df.columns and 'overall_satisfaction' in services_df.columns:
            parts_impact = services_df.groupby('parts_availability')['overall_satisfaction'].mean()
            
            fig_parts = px.bar(
                x=parts_impact.index,
                y=parts_impact.values,
                title="Parts Availability Impact on Satisfaction",
                labels={'x': 'Parts Availability', 'y': 'Average Satisfaction'},
                color=parts_impact.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_parts, use_container_width=True)
        else:
            st.info("Parts availability data not available")

def customer_insights(customers_df, services_df):
    st.header("Customer Insights Dashboard")
    
    # Customer Demographics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'age' in customers_df.columns:
            avg_age = customers_df['age'].mean()
            st.metric("Average Age", f"{avg_age:.0f} years")
        else:
            st.metric("Average Age", "N/A")
    
    with col2:
        if 'annual_income' in customers_df.columns:
            avg_income = customers_df['annual_income'].mean()
            st.metric("Average Income", f"₹{avg_income:,.0f}")
        else:
            st.metric("Average Income", "N/A")
    
    with col3:
        if 'loyalty_score' in customers_df.columns:
            avg_loyalty = customers_df['loyalty_score'].mean()
            st.metric("Average Loyalty", f"{avg_loyalty:.2f}")
        else:
            st.metric("Average Loyalty", "N/A")
    
    with col4:
        if 'annual_income' in customers_df.columns:
            premium_customers = (customers_df['annual_income'] > 1000000).sum()
            st.metric("Premium Customers", f"{premium_customers:,}")
        else:
            st.metric("Premium Customers", "N/A")
    
    # Customer Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution
        if 'age' in customers_df.columns:
            fig_age = px.histogram(
                customers_df,
                x='age',
                nbins=20,
                title="Customer Age Distribution",
                labels={'age': 'Age', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age distribution data not available")
    
    with col2:
        # Income vs Loyalty
        if 'annual_income' in customers_df.columns and 'loyalty_score' in customers_df.columns:
            fig_income_loyalty = px.scatter(
                customers_df.sample(min(1000, len(customers_df))),
                x='annual_income',
                y='loyalty_score',
                color='city' if 'city' in customers_df.columns else None,
                title="Income vs Loyalty Score",
                labels={'annual_income': 'Annual Income (₹)', 'loyalty_score': 'Loyalty Score'}
            )
            st.plotly_chart(fig_income_loyalty, use_container_width=True)
        else:
            st.info("Income vs loyalty data not available")
    
    # Brand Preferences
    col1, col2 = st.columns(2)
    
    with col1:
        # Car Brand Distribution
        if 'car_brand' in customers_df.columns:
            brand_dist = customers_df['car_brand'].value_counts()
            fig_brands = px.bar(
                x=brand_dist.values,
                y=brand_dist.index,
                orientation='h',
                title="Car Brand Distribution",
                labels={'x': 'Number of Customers', 'y': 'Car Brand'}
            )
            st.plotly_chart(fig_brands, use_container_width=True)
        else:
            st.info("Car brand data not available")
    
    with col2:
        # Customer Segmentation by Income
        if 'annual_income' in customers_df.columns:
            income_segments = pd.cut(customers_df['annual_income'],
                                   bins=[0, 500000, 1000000, 1500000, float('inf')],
                                   labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
            segment_dist = income_segments.value_counts()
            
            fig_segments = px.pie(
                values=segment_dist.values,
                names=segment_dist.index,
                title="Customer Income Segments"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        else:
            st.info("Income segmentation data not available")
    
    # Customer Journey Analysis
    st.subheader("Customer Journey Analysis")
    
    if 'customer_id' in services_df.columns and 'customer_id' in customers_df.columns:
        # Merge customer and service data for journey analysis
        customer_journey = services_df.merge(customers_df, on='customer_id')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Satisfaction by Customer Segment
            if 'annual_income' in customer_journey.columns and 'overall_satisfaction' in customer_journey.columns:
                income_segments = pd.cut(customer_journey['annual_income'],
                                       bins=[0, 500000, 1000000, 1500000, float('inf')],
                                       labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
                segment_satisfaction = customer_journey.groupby(income_segments)['overall_satisfaction'].mean()
                
                fig_segment_sat = px.bar(
                    x=segment_satisfaction.index,
                    y=segment_satisfaction.values,
                    title="Satisfaction by Customer Segment",
                    labels={'x': 'Income Segment', 'y': 'Average Satisfaction'}
                )
                st.plotly_chart(fig_segment_sat, use_container_width=True)
            else:
                st.info("Segment satisfaction data not available")
        
        with col2:
            # Communication Preference Analysis
            if 'preferred_communication' in customer_journey.columns and 'overall_satisfaction' in customer_journey.columns:
                comm_satisfaction = customer_journey.groupby('preferred_communication')['overall_satisfaction'].mean()
                
                fig_comm = px.bar(
                    x=comm_satisfaction.index,
                    y=comm_satisfaction.values,
                    title="Satisfaction by Communication Preference",
                    labels={'x': 'Communication Method', 'y': 'Average Satisfaction'}
                )
                st.plotly_chart(fig_comm, use_container_width=True)
            else:
                st.info("Communication preference data not available")
    else:
        st.info("Customer journey analysis requires customer and service data to be linked")

def service_center_performance(services_df, centers_df):
    st.header("Service Center Performance Dashboard")
    
    if 'service_center' not in services_df.columns:
        st.warning("Service center data not available in the dataset")
        return
    
    # Performance Metrics
    center_performance = services_df.groupby('service_center').agg({
        'overall_satisfaction': 'mean' if 'overall_satisfaction' in services_df.columns else 'count',
        'delay_hours': 'mean' if 'delay_hours' in services_df.columns else 'count',
        'total_cost': ['sum', 'mean'] if 'total_cost' in services_df.columns else 'count',
        'service_id': 'count'
    }).round(2)
    
    # Flatten column names
    if 'total_cost' in services_df.columns and 'overall_satisfaction' in services_df.columns:
        center_performance.columns = ['Avg_Satisfaction', 'Avg_Delay', 'Total_Revenue', 'Avg_Cost', 'Service_Count']
    else:
        center_performance.columns = [f'Metric_{i}' for i in range(len(center_performance.columns))]
    
    center_performance = center_performance.reset_index()
    
    # Top/Bottom Performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Centers (Satisfaction)")
        if 'Avg_Satisfaction' in center_performance.columns:
            top_centers = center_performance.nlargest(5, 'Avg_Satisfaction')[['service_center', 'Avg_Satisfaction', 'Service_Count']]
            st.dataframe(top_centers, use_container_width=True)
        else:
            st.info("Satisfaction data not available")
    
    with col2:
        st.subheader("Centers Needing Improvement")
        if 'Avg_Satisfaction' in center_performance.columns:
            bottom_centers = center_performance.nsmallest(5, 'Avg_Satisfaction')[['service_center', 'Avg_Satisfaction'] + (['Avg_Delay'] if 'Avg_Delay' in center_performance.columns else [])]
            st.dataframe(bottom_centers, use_container_width=True)
        else:
            st.info("Performance improvement data not available")
    
    # Performance Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction vs Revenue
        if 'Total_Revenue' in center_performance.columns and 'Avg_Satisfaction' in center_performance.columns:
            fig_perf = px.scatter(
                center_performance,
                x='Total_Revenue',
                y='Avg_Satisfaction',
                size='Service_Count',
                hover_name='service_center',
                title="Revenue vs Satisfaction by Center",
                labels={'Total_Revenue': 'Total Revenue (₹)', 'Avg_Satisfaction': 'Average Satisfaction'}
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Revenue vs satisfaction data not available")
    
    with col2:
        # Delay Analysis
        if 'Avg_Delay' in center_performance.columns:
            top_delay_centers = center_performance.nlargest(10, 'Avg_Delay')
            fig_delay = px.bar(
                top_delay_centers,
                x='service_center',
                y='Avg_Delay',
                title="Centers with Highest Average Delays",
                labels={'service_center': 'Service Center', 'Avg_Delay': 'Average Delay (hours)'}
            )
            fig_delay.update_xaxes(tickangle=45)
            st.plotly_chart(fig_delay, use_container_width=True)
        else:
            st.info("Delay analysis data not available")
    
    # Detailed Center Analysis
    st.subheader("Detailed Center Analysis")
    
    selected_center = st.selectbox(
        "Select Service Center for Detailed Analysis",
        options=center_performance['service_center'].unique()
    )
    
    if selected_center:
        center_data = services_df[services_df['service_center'] == selected_center]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Services", len(center_data))
        with col2:
            if 'overall_satisfaction' in center_data.columns:
                st.metric("Avg Satisfaction", f"{center_data['overall_satisfaction'].mean():.2f}")
            else:
                st.metric("Avg Satisfaction", "N/A")
        with col3:
            if 'total_cost' in center_data.columns:
                st.metric("Total Revenue", f"₹{center_data['total_cost'].sum():,.0f}")
            else:
                st.metric("Total Revenue", "N/A")
        with col4:
            if 'delay_hours' in center_data.columns:
                st.metric("Avg Delay", f"{center_data['delay_hours'].mean():.1f} hrs")
            else:
                st.metric("Avg Delay", "N/A")
        
        # Center-specific charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Service type performance
            if 'service_type' in center_data.columns and 'overall_satisfaction' in center_data.columns:
                service_perf = center_data.groupby('service_type')['overall_satisfaction'].mean().sort_values(ascending=True)
                
                fig_service = px.bar(
                    x=service_perf.values,
                    y=service_perf.index,
                    orientation='h',
                    title=f"Service Type Performance - {selected_center}",
                    labels={'x': 'Average Satisfaction', 'y': 'Service Type'}
                )
                st.plotly_chart(fig_service, use_container_width=True)
            else:
                st.info("Service type performance data not available")
        
        with col2:
            # Monthly trend
            if 'service_date' in center_data.columns and 'overall_satisfaction' in center_data.columns:
                monthly_trend = center_data.groupby(center_data['service_date'].dt.to_period('M'))['overall_satisfaction'].mean()
                
                if len(monthly_trend) > 0:
                    fig_trend = px.line(
                        x=monthly_trend.index.astype(str),
                        y=monthly_trend.values,
                        title=f"Satisfaction Trend - {selected_center}",
                        labels={'x': 'Month', 'y': 'Average Satisfaction'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No trend data available for selected period")
            else:
                st.info("Monthly trend data not available")

def feedback_analysis(feedback_df, services_df):
    st.header("Feedback & Complaints Analysis")
    
    if len(feedback_df) == 0:
        st.warning("No feedback data available for the selected filters.")
        return
    
    # Feedback Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_feedback = len(feedback_df)
        st.metric("Total Feedback", f"{total_feedback:,}")
    
    with col2:
        if 'resolution_status' in feedback_df.columns:
            resolved_rate = (feedback_df['resolution_status'] == 'Resolved').mean() * 100
            st.metric("Resolution Rate", f"{resolved_rate:.1f}%")
        else:
            st.metric("Resolution Rate", "N/A")
    
    with col3:
        if 'resolution_days' in feedback_df.columns:
            avg_resolution_days = feedback_df[feedback_df['resolution_status'] == 'Resolved']['resolution_days'].mean()
            if not pd.isna(avg_resolution_days):
                st.metric("Avg Resolution Time", f"{avg_resolution_days:.1f} days")
            else:
                st.metric("Avg Resolution Time", "N/A")
        else:
            st.metric("Avg Resolution Time", "N/A")
    
    with col4:
        if 'escalation_required' in feedback_df.columns:
            escalation_rate = feedback_df['escalation_required'].mean() * 100
            st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
        else:
            st.metric("Escalation Rate", "N/A")
    
    # Feedback Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Complaint Categories
        if 'complaint_category' in feedback_df.columns:
            complaint_dist = feedback_df['complaint_category'].value_counts()
            
            fig_complaints = px.bar(
                x=complaint_dist.values,
                y=complaint_dist.index,
                orientation='h',
                title="Complaint Categories",
                labels={'x': 'Number of Complaints', 'y': 'Complaint Category'}
            )
            st.plotly_chart(fig_complaints, use_container_width=True)
        else:
            st.info("Complaint category data not available")
    
    with col2:
        # Resolution Status
        if 'resolution_status' in feedback_df.columns:
            resolution_dist = feedback_df['resolution_status'].value_counts()
            
            fig_resolution = px.pie(
                values=resolution_dist.values,
                names=resolution_dist.index,
                title="Resolution Status Distribution"
            )
            st.plotly_chart(fig_resolution, use_container_width=True)
        else:
            st.info("Resolution status data not available")
    
    # Feedback Channel Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Feedback Channel Effectiveness
        if 'feedback_channel' in feedback_df.columns and 'resolution_status' in feedback_df.columns:
            channel_resolution = feedback_df.groupby('feedback_channel')['resolution_status'].apply(
                lambda x: (x == 'Resolved').mean() * 100
            ).sort_values(ascending=True)
            
            fig_channel = px.bar(
                x=channel_resolution.values,
                y=channel_resolution.index,
                orientation='h',
                title="Resolution Rate by Feedback Channel",
                labels={'x': 'Resolution Rate (%)', 'y': 'Feedback Channel'}
            )
            st.plotly_chart(fig_channel, use_container_width=True)
        else:
            st.info("Channel effectiveness data not available")
    
    with col2:
        # Resolution Time by Category
        if 'complaint_category' in feedback_df.columns and 'resolution_days' in feedback_df.columns:
            resolution_time_by_category = feedback_df[feedback_df['resolution_days'].notna()].groupby('complaint_category')['resolution_days'].mean().sort_values(ascending=True)
            
            fig_resolution_time = px.bar(
                x=resolution_time_by_category.values,
                y=resolution_time_by_category.index,
                orientation='h',
                title="Average Resolution Time by Category",
                labels={'x': 'Average Resolution Days', 'y': 'Complaint Category'}
            )
            st.plotly_chart(fig_resolution_time, use_container_width=True)
        else:
            st.info("Resolution time data not available")
    
    # Detailed Feedback Table
    st.subheader("Recent Feedback Details")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    filter_columns = ['All']
    if 'resolution_status' in feedback_df.columns:
        filter_columns.extend(list(feedback_df['resolution_status'].unique()))
    
    with col1:
        status_filter = st.selectbox("Filter by Status", filter_columns)
    
    category_columns = ['All']
    if 'complaint_category' in feedback_df.columns:
        category_columns.extend(list(feedback_df['complaint_category'].unique()))
    
    with col2:
        category_filter = st.selectbox("Filter by Category", category_columns)
    
    channel_columns = ['All']
    if 'feedback_channel' in feedback_df.columns:
        channel_columns.extend(list(feedback_df['feedback_channel'].unique()))
    
    with col3:
        channel_filter = st.selectbox("Filter by Channel", channel_columns)
    
    # Apply filters
    filtered_feedback = feedback_df.copy()
    
    if status_filter != 'All' and 'resolution_status' in filtered_feedback.columns:
        filtered_feedback = filtered_feedback[filtered_feedback['resolution_status'] == status_filter]
    if category_filter != 'All' and 'complaint_category' in filtered_feedback.columns:
        filtered_feedback = filtered_feedback[filtered_feedback['complaint_category'] == category_filter]
    if channel_filter != 'All' and 'feedback_channel' in filtered_feedback.columns:
        filtered_feedback = filtered_feedback[filtered_feedback['feedback_channel'] == channel_filter]
    
    # Display filtered data
    available_columns = [col for col in ['feedback_id', 'complaint_category', 'resolution_status', 
                      'resolution_days', 'feedback_channel', 'escalation_required'] if col in filtered_feedback.columns]
    
    if available_columns:
        st.dataframe(filtered_feedback[available_columns].head(20), use_container_width=True)
    else:
        st.info("No suitable columns available for display")

def predictive_analytics(services_df, customers_df):
    st.header("Predictive Analytics Dashboard")
    
    # Customer Satisfaction Prediction Factors
    st.subheader("Key Factors Affecting Customer Satisfaction")
    
    # Correlation Analysis
    if 'overall_satisfaction' in services_df.columns:
        numeric_cols = []
        potential_cols = ['actual_duration_hours', 'delay_hours', 'total_cost', 'wait_time_minutes', 
                         'technician_skill_rating', 'service_advisor_rating', 'facility_cleanliness_rating']
        
        for col in potential_cols:
            if col in services_df.columns:
                numeric_cols.append(col)
        
        if numeric_cols:
            correlation_data = services_df[numeric_cols + ['overall_satisfaction']].corr()['overall_satisfaction'].drop('overall_satisfaction')
            correlation_data = correlation_data.sort_values(key=abs, ascending=True)
            
            fig_corr = px.bar(
                x=correlation_data.values,
                y=correlation_data.index,
                orientation='h',
                title="Correlation with Customer Satisfaction",
                labels={'x': 'Correlation Coefficient', 'y': 'Factors'},
                color=correlation_data.values,
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient numeric data for correlation analysis")
    else:
        st.info("Customer satisfaction data not available")
    
    # Customer Lifetime Value Analysis
    st.subheader("Customer Lifetime Value Analysis")
    
    if 'customer_id' in services_df.columns:
        # Calculate CLV metrics
        clv_metrics = {}
        
        if 'total_cost' in services_df.columns:
            clv_metrics['Total_Spent'] = 'sum'
        if 'service_id' in services_df.columns:
            clv_metrics['Service_Count'] = 'count'
        if 'overall_satisfaction' in services_df.columns:
            clv_metrics['Avg_Satisfaction'] = 'mean'
        if 'service_date' in services_df.columns:
            clv_metrics['First_Service'] = 'min'
            clv_metrics['Last_Service'] = 'max'
        
        if clv_metrics:
            customer_clv = services_df.groupby('customer_id').agg({
                'total_cost': clv_metrics.get('Total_Spent', 'count'),
                'service_id': clv_metrics.get('Service_Count', 'count'),
                'overall_satisfaction': clv_metrics.get('Avg_Satisfaction', 'mean'),
                'service_date': [clv_metrics.get('First_Service', 'min'), clv_metrics.get('Last_Service', 'max')] if 'service_date' in services_df.columns else 'count'
            })
            
            # Flatten column names
            if 'service_date' in services_df.columns:
                customer_clv.columns = ['Total_Spent', 'Service_Count', 'Avg_Satisfaction', 'First_Service', 'Last_Service']
                customer_clv['Days_Active'] = (customer_clv['Last_Service'] - customer_clv['First_Service']).dt.days + 1
            else:
                customer_clv.columns = ['Total_Spent', 'Service_Count', 'Avg_Satisfaction']
            
            if 'Total_Spent' in customer_clv.columns and 'Service_Count' in customer_clv.columns:
                customer_clv['Avg_Service_Value'] = customer_clv['Total_Spent'] / customer_clv['Service_Count']
            
            # Merge with customer data if available
            if 'customer_id' in customers_df.columns:
                merge_cols = [col for col in ['annual_income', 'loyalty_score', 'car_brand'] if col in customers_df.columns]
                if merge_cols:
                    customer_clv = customer_clv.merge(
                        customers_df.set_index('customer_id')[merge_cols], 
                        left_index=True, right_index=True, how='left'
                    )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CLV Distribution
                if 'Total_Spent' in customer_clv.columns:
                    fig_clv = px.histogram(
                        customer_clv,
                        x='Total_Spent',
                        nbins=30,
                        title="Customer Lifetime Value Distribution",
                        labels={'Total_Spent': 'Total Amount Spent (₹)', 'count': 'Number of Customers'}
                    )
                    st.plotly_chart(fig_clv, use_container_width=True)
                else:
                    st.info("Customer lifetime value data not available")
            
            with col2:
                # Service Frequency vs Value
                if 'Service_Count' in customer_clv.columns and 'Total_Spent' in customer_clv.columns:
                    color_col = 'Avg_Satisfaction' if 'Avg_Satisfaction' in customer_clv.columns else None
                    size_col = 'annual_income' if 'annual_income' in customer_clv.columns else None
                    
                    fig_freq_value = px.scatter(
                        customer_clv.sample(min(1000, len(customer_clv))),
                        x='Service_Count',
                        y='Total_Spent',
                        color=color_col,
                        size=size_col,
                        title="Service Frequency vs Customer Value",
                        labels={'Service_Count': 'Number of Services', 'Total_Spent': 'Total Spent (₹)'}
                    )
                    st.plotly_chart(fig_freq_value, use_container_width=True)
                else:
                    st.info("Service frequency vs value data not available")
        else:
            st.info("Insufficient data for customer lifetime value analysis")
    else:
        st.info("Customer ID data not available")
    
    # Customer Segmentation
    st.subheader("Customer Segmentation Analysis")
    
    if 'customer_id' in services_df.columns and 'service_date' in services_df.columns:
        # RFM Analysis (Recency, Frequency, Monetary)
        current_date = services_df['service_date'].max()
        
        rfm_metrics = {}
        if 'service_date' in services_df.columns:
            rfm_metrics['Recency'] = lambda x: (current_date - x.max()).days
        if 'service_id' in services_df.columns:
            rfm_metrics['Frequency'] = 'count'  
        if 'total_cost' in services_df.columns:
            rfm_metrics['Monetary'] = 'sum'
        
        if rfm_metrics:
            rfm = services_df.groupby('customer_id').agg({
                'service_date': rfm_metrics.get('Recency', 'max'),
                'service_id': rfm_metrics.get('Frequency', 'count'),
                'total_cost': rfm_metrics.get('Monetary', 'sum')
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            
            # Create RFM scores
            rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5,4,3,2,1])
            rfm['F_Score'] = pd.cut(rfm['Frequency'].rank(method='first'), bins=5, labels=[1,2,3,4,5])
            rfm['M_Score'] = pd.cut(rfm['Monetary'].rank(method='first'), bins=5, labels=[1,2,3,4,5])
            
            rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
            
            # Define customer segments
            def segment_customers(rfm_score):
                if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
                    return 'Champions'
                elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                    return 'Loyal Customers'
                elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
                    return 'New Customers'
                elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
                    return 'At Risk'
                elif rfm_score in ['111', '112', '121', '131', '141', '151']:
                    return 'Lost Customers'
                else:
                    return 'Others'
            
            rfm['Segment'] = rfm['RFM_Score'].apply(segment_customers)
            
            # Visualize segments
            col1, col2 = st.columns(2)
            
            with col1:
                segment_dist = rfm['Segment'].value_counts()
                fig_segments = px.pie(
                    values=segment_dist.values,
                    names=segment_dist.index,
                    title="Customer Segments Distribution"
                )
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                segment_value = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=True)
                fig_segment_value = px.bar(
                    x=segment_value.values,
                    y=segment_value.index,
                    orientation='h',
                    title="Average Monetary Value by Segment",
                    labels={'x': 'Average Monetary Value (₹)', 'y': 'Customer Segment'}
                )
                st.plotly_chart(fig_segment_value, use_container_width=True)
        else:
            st.info("Insufficient data for RFM analysis")
    else:
        st.info("Customer segmentation requires customer ID and service date data")
    
    # Churn Risk Prediction
    st.subheader("Churn Risk Analysis")
    
    if 'customer_id' in services_df.columns and 'service_date' in services_df.columns:
        # Calculate days since last service
        current_date = services_df['service_date'].max()
        last_service = services_df.groupby('customer_id')['service_date'].max()
        days_since_last = (current_date - last_service).dt.days
        
        # Define churn risk categories
        def churn_risk(days):
            if days <= 90:
                return 'Low Risk'
            elif days <= 180:
                return 'Medium Risk'
            elif days <= 365:
                return 'High Risk'
            else:
                return 'Very High Risk'
        
        churn_analysis = pd.DataFrame({
            'customer_id': days_since_last.index,
            'days_since_last_service': days_since_last.values,
            'churn_risk': days_since_last.apply(churn_risk)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_dist = churn_analysis['churn_risk'].value_counts()
            fig_churn = px.bar(
                x=churn_dist.index,
                y=churn_dist.values,
                title="Churn Risk Distribution",
                labels={'x': 'Churn Risk Category', 'y': 'Number of Customers'},
                color=churn_dist.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            # Service demand trend
            if len(services_df) > 0:
                monthly_services = services_df.groupby(services_df['service_date'].dt.to_period('M')).size()
                
                if len(monthly_services) > 0:
                    fig_trend = px.line(
                        x=monthly_services.index.astype(str),
                        y=monthly_services.values,
                        title="Monthly Service Volume Trend",
                        labels={'x': 'Month', 'y': 'Number of Services'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No service trend data available")
            else:
                st.info("Service demand data not available")
    else:
        st.info("Churn analysis requires customer ID and service date data")
    
    # Actionable Insights
    st.subheader("Actionable Insights & Recommendations")
    
    insights = []
    
    # Service quality insights
    if 'overall_satisfaction' in services_df.columns and 'service_type' in services_df.columns:
        high_satisfaction_services = services_df[services_df['overall_satisfaction'] >= 4.5]
        if len(high_satisfaction_services) > 0:
            best_service_type = high_satisfaction_services['service_type'].mode()
            if len(best_service_type) > 0:
                insights.append(f"'{best_service_type.iloc[0]}' has the highest customer satisfaction rate")
    
    # Delay insights
    if 'delay_hours' in services_df.columns and 'service_type' in services_df.columns:
        high_delay_services = services_df[services_df['delay_hours'] > 2.0]
        if len(high_delay_services) > 0:
            worst_delay_service = high_delay_services.groupby('service_type')['delay_hours'].mean().idxmax()
            insights.append(f"'{worst_delay_service}' services have the highest average delays - needs process improvement")
    
    # Revenue insights
    if 'service_type' in services_df.columns and 'total_cost' in services_df.columns:
        high_revenue_services = services_df.groupby('service_type')['total_cost'].sum().idxmax()
        insights.append(f"'{high_revenue_services}' generates the highest revenue - consider promotional campaigns")
    
    # Display insights
    if insights:
        for i, insight in enumerate(insights, 1):
            st.info(f"{i}. {insight}")
    else:
        st.info("Insufficient data to generate specific insights")
    
    # Recommendations table
    st.subheader("Strategic Recommendations")
    
    recommendations = pd.DataFrame({
        'Priority': ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Area': ['Service Quality', 'Customer Retention', 'Operational Efficiency', 'Revenue Growth', 'Technology'],
        'Recommendation': [
            'Focus on technician training for services with low satisfaction scores',
            'Implement targeted retention campaigns for at-risk high-value customers',
            'Optimize scheduling to reduce wait times and delays',
            'Promote high-margin services to increase average transaction value',
            'Enhance digital communication channels for tech-savvy customers'
        ],
        'Expected Impact': ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Timeline': ['3 months', '1 month', '2 months', '6 months', '4 months']
    })
    
    st.dataframe(recommendations, use_container_width=True)

if __name__ == "__main__":
    main()