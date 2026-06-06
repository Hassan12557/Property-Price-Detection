import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. HCI Spatial Setting: Maximize sensory screen real estate layout
st.set_page_config(
    page_title="Zameen Advanced Valuation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SYSTEM DIRECTORY PATH CONFIGURATIONS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "zameen_train_preprocessed.csv")


@st.cache_resource
def load_production_ml_core():
    """
    Optimizes system memory footprints by caching model binaries.
    Prevents repeated disk I/O bottlenecks during runtime execution loops.
    """
    pipeline_asset_path = os.path.join(MODELS_DIR, "data_pipeline_assets.joblib")
    model_checkpoint_path = os.path.join(MODELS_DIR, "model.pkl")

    if not os.path.exists(pipeline_asset_path) or not os.path.exists(model_checkpoint_path):
        return None, None

    return joblib.load(pipeline_asset_path), joblib.load(model_checkpoint_path)


@st.cache_data
def load_analytics_dataset():
    """ Loads preprocessed dataset on-demand for analytics tab visualization. """
    if os.path.exists(PROCESSED_DATA_PATH):
        return pd.read_csv(PROCESSED_DATA_PATH)
    return None


# Load underlying system frameworks
pipeline_assets, model = load_production_ml_core()
analytics_df = load_analytics_dataset()

if pipeline_assets is None or model is None:
    st.error("🚨 System Core Core Error: Target Weights or Scaler Objects Missing inside models/ directory.")
    st.stop()

# Deconstruct serialization mapping dictionaries
feature_columns = pipeline_assets["feature_columns"]
location_map = pipeline_assets["location_map"]
global_mean = pipeline_assets["global_mean"]
scaler = pipeline_assets["scaler"]
known_locations = sorted(list(location_map.keys()))

# --- SIDEBAR CONTROL ANCHOR (UX Law of Proximity / System Status Global Controls) ---
with st.sidebar:
    st.markdown("<h2 style='color:#1E3A8A;'>⚙️ System Diagnostics</h2>", unsafe_with_html=True)
    st.success("✅ Model Core Connected")
    st.info(f"Loaded Features: {len(feature_columns)} matrix inputs")

    st.markdown("---")
    st.markdown("### 🏢 Quick-Set Presets")
    st.caption("Resets the system parameters to standardized baseline profiles to ease interaction entry loops.")
    preset_selection = st.selectbox("Market Profiles",
                                    ["Manual Input", "Premium Islamabad Estate", "Standard Lahore Plot"])

# Set macro preset values dynamically to manage Intrinsic Cognitive Load bounds
if preset_selection == "Premium Islamabad Estate":
    default_beds, default_baths, default_area, default_unit, default_city, default_loc = 5, 5, 1.0, "Kanal", "Islamabad", "DHA Defence"
elif preset_selection == "Standard Lahore Plot":
    default_beds, default_baths, default_area, default_unit, default_city, default_loc = 3, 3, 5.0, "Marla", "Lahore", "DHA Defence"
else:
    default_beds, default_baths, default_area, default_unit, default_city, default_loc = 3, 3, 5.0, "Marla", "Islamabad", "G-10"

# --- WORKSPACE TAB SEGREGATION ARCHITECTURE (PACT Separation Framework) ---
tab_app, tab_analytics = st.tabs(["🔮 Real-Time Valuation Engine", "📊 Market Analytics Workspace"])

# =============================================================================
# TAB 1: OPERATIONAL REAL-TIME VALUATION INTERFACE
# =============================================================================
with tab_app:
    st.markdown("<h1 style='color: #1E3A8A;'>⚡ Property Price Valuation Engine</h1>", unsafe_with_html=True)
    st.markdown(
        "Provide structural profile criteria below. Tree-based predictive pipelines compute estimates automatically.")

    # Grouping features into layout modules to reduce scanning friction
    with st.form(key="advanced_inference_form"):
        st.markdown("#### 📐 Structural Geometry & Target Category")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            area_size = st.number_input("Property Area Size Scalar", min_value=0.5, max_value=500.0, value=default_area,
                                        step=0.5)
        with r1_c2:
            area_type = st.selectbox("Unit Matrix Designation", options=["Marla", "Kanal"],
                                     index=0 if default_unit == "Marla" else 1)
        with r1_c3:
            property_type = st.selectbox("Structural Classification Typology",
                                         options=["House", "Flat", "Upper Portion", "Lower Portion"])

        st.markdown("---")
        st.markdown("#### 🛏️ Internal Layout Composition (Gestalt Proximity Map)")
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            bedrooms = st.slider("Active Bedroom Count Capacity", min_value=1, max_value=12, value=default_beds)
        with r2_c2:
            bathrooms = st.slider("Active Bathroom Count Capacity", min_value=1, max_value=12, value=default_baths)

        st.markdown("---")
        st.markdown("#### 🗺️ Regional Boundary Domain Assignment")
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1:
            city = st.selectbox("Metropolitan Urban Grid Target",
                                options=["Islamabad", "Lahore", "Karachi", "Rawalpindi", "Faisalabad"],
                                index=["Islamabad", "Lahore", "Karachi", "Rawalpindi", "Faisalabad"].index(
                                    default_city))
        with r3_c2:
            location = st.selectbox("Searchable Neighborhood Sector Node", options=known_locations,
                                    index=known_locations.index(default_loc) if default_loc in known_locations else 0)
        with r3_c3:
            province = st.selectbox("Macro Provincial Governance Realm",
                                    options=["Islamabad Capital Territory", "Punjab", "Sindh", "Khyber Pakhtunkhwa"])

        with st.expander("🌐 Advanced High-Fidelity Spatial Coordinates (GPS Overrides)"):
            r4_c1, r4_c2 = st.columns(2)
            with r4_c1:
                latitude = st.number_input("Geographic Latitude Node Coordinate", format="%.6f", value=33.6844)
            with r4_c2:
                longitude = st.number_input("Geographic Longitude Node Coordinate", format="%.6f", value=73.0479)

        # Fitts's Law Optimized Form Activation Target Action Link
        st.markdown("<br>", unsafe_with_html=True)
        compute_valuation = st.form_submit_button(label="🚀 COMPUTE SYSTEM INTERACTION INFERENCE",
                                                  use_container_width=True)

    # --- POST-SUBMIT PROCESSING & ISOLATED VISUAL INTERACTION GRAPHICS ---
    if compute_valuation:
        # Convert area dimension context math
        converted_sqft = area_size * 4500 if area_type == "Kanal" else area_size * 225

        # Structure payload package row
        query_payload = {
            "bedrooms": float(bedrooms), "bathroom": float(bathrooms), "area_sqft": float(converted_sqft),
            "latitude": float(latitude), "longitude": float(longitude),
            "property_type": property_type, "city": city, "province_name": province
        }

        # Replicate feature engine processing path pipelines
        eval_df = pd.DataFrame([query_payload])
        eval_df["location_encoded"] = location_map.get(location, global_mean)
        eval_df_enc = pd.get_dummies(eval_df, columns=["property_type", "city", "province_name"])
        eval_df_final = eval_df_enc.reindex(columns=feature_columns, fill_value=0)

        # Normalize continuous scale attributes safely
        scale_cols = ["bedrooms", "bathroom", "area_sqft", "latitude", "longitude"]
        scale_cols = [c for c in scale_cols if c in eval_df_final.columns]
        eval_df_final[scale_cols] = scaler.transform(eval_df_final[scale_cols])

        # Execute forward prediction calculation
        predicted_log_price = model.predict(eval_df_final)[0]
        calculated_pkr_price = np.expm1(predicted_log_price)

        # --- THE LAW OF ISOLATION DISPLAY CARD ---
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 30px; border-radius: 16px; text-align: center; color: white; box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4); margin-top: 20px;">
                <h3 style="margin: 0; font-weight: 400; text-transform: uppercase; letter-spacing: 2px; font-size: 1.1em; color: #E0E7FF;">Computed Evaluation Valuation</h3>
                <h1 style="margin: 15px 0 5px 0; font-size: 3.2em; font-weight: 800; letter-spacing: -1px;">PKR {calculated_pkr_price:,.2f}</h1>
                <p style="margin: 0; font-size: 1.3em; font-weight: 300; color: #93C5FD;">≈ <b>{calculated_pkr_price / 10000000:.2f} Crore</b> PKR &nbsp;|&nbsp; <b>{calculated_pkr_price / 1000000:.2f} Million</b> PKR</p>
            </div>
        """, unsafe_with_html=True)

        # Add a localized context gauge comparing target to neighborhood bounds
        st.markdown("<br>", unsafe_with_html=True)
        loc_base_value = np.expm1(location_map.get(location, global_mean))

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=calculated_pkr_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Price Level Compared to {location} Regional Baseline Average", 'font': {'size': 16}},
            delta={'reference': loc_base_value, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
            gauge={
                'axis': {'range': [None, max(calculated_pkr_price * 1.5, loc_base_value * 1.5)]},
                'bar': {'color': "#1E3A8A"},
                'steps': [
                    {'range': [0, loc_base_value], 'color': '#F3F4F6'},
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# =============================================================================
# TAB 2: ADVANCED SYSTEM ANALYTICS WORKSPACE
# =============================================================================
with tab_analytics:
    st.markdown("<h1 style='color: #1E3A8A;'>📊 Real Estate Market Analytics Dashboard</h1>", unsafe_with_html=True)
    st.markdown("Visualizing empirical trends extracted straight from the preprocessed feature matrix layers.")

    if analytics_df is not None:
        # Macro Overview Metric Layout Strips
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Profile Rows", f"{len(analytics_df):,}")
        with m2:
            avg_log_p = analytics_df["target_log_price"].mean()
            st.metric("Median Market Target", f"PKR {np.expm1(avg_log_p):,.0f}")
        with m3:
            st.metric("Average Bedrooms Matrix", f"{analytics_df['bedrooms'].mean():.1f} Rooms")
        with m4:
            st.metric("Unified Data Dimensionality", f"{analytics_df.shape[1]} Engine Attributes")

        st.markdown("<br>", unsafe_with_html=True)

        # Interactive Layout Grid
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown("#### 📈 Target Value Symmetrization Curve")
            # Dynamic visualization loop
            fig_hist = px.histogram(
                analytics_df, x="target_log_price", kde=True,
                labels={"target_log_price": "Log Price Scalar Bounds"},
                color_discrete_sequence=["#1E3A8A"]
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with graph_col2:
            st.markdown("#### 📐 Structural Footprint Elasticity")
            # Prevent browser rendering delay limits by downsampling interactive rows safely
            sample_size = min(2000, len(analytics_df))
            sample_data = analytics_df.sample(n=sample_size, random_state=42)

            fig_scatter = px.scatter(
                sample_data, x="area_sqft", y="target_log_price",
                trendline="ols", trendline_color_override="#ef4444",
                labels={"area_sqft": "Scaled Area (Standardized)", "target_log_price": "Log Price"},
                opacity=0.5, color_discrete_sequence=["#3b82f6"]
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.warning("⚠️ Analytics Workspace Sleeping: Unable to find `zameen_train_preprocessed.csv` data source loops.")
        st.info("Ensure files are generated at `Data/Processed/` to illuminate analytics graphs.")