import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

st.set_page_config(page_title="NVIDIA GPU Analysis", page_icon="⚡", layout="wide")
st.title("NVIDIA GPU Dataset Analysis")
st.write("Discover the performance and specs between 40 and 50 series Gpus.")

#Data 
@st.cache_data
def load_data():
    data = {
        "Model": ["RTX 4050(Laptop)", "RTX 4060","RTX 4070", "RTX 4070 Ti", "RTX 4080", "RTX 4090", "RTX 5050", "RTX 5060", "RTX 5070", "RTX 5070 Ti", "RTX 5080", "RTX 5090"],
        "Series": ["40 Series", "40 Series", "40 Series", "40 Series", "40 Series", "40 Series", "50 Series", "50 Series", "50 Series", "50 Series", "50 Series", "50 Series"],
        "Architecture": ["Ada Lovelace", "Ada Lovelace", "Ada Lovelace", "Ada Lovelace", "Ada Lovelace", "Ada Lovelace", "Blackwell", "Blackwell", "Blackwell", "Blackwell", "Blackwell", "Blackwell"],
        "VRAM_GB": [6, 8, 12, 12, 16, 24, 8, 8, 12, 16, 16, 32],
        "Memory_Type": ["GDDR6", "GDDR6", "GDDR6X", "GDDR6X", "GDDR6X", "GDDR6X", "GDDR6", "GDDR7", "GDDR7", "GDDR7", "GDDR7", "GDDR7"],
        "TDP(Watts)": [140, 115, 200, 285, 320, 450, 130, 145, 250, 300, 400, 575],
        "Compute_TFLOPS": [12.1, 15.11, 29.1, 40.1, 48.7, 82.6, 13.6, 19.2, 30.8, 43.9, 56.3, 104.9],
        "Gaming_4K_FPS": [28, 40, 65, 78, 95, 125, 37, 48, 75, 92, 110, 165]
    }
    return pd.DataFrame(data)

try:
    df = load_data()
except Exception:
    st.error("Dataset could not be loaded.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter GPUs")

if "Series" in df.columns:
    series_options = sorted(df["Series"].dropna().unique())
    selected_series = st.sidebar.multiselect(
        "Select GPU Series",
        options=series_options,
        default=series_options
    )
    df = df[df["Series"].isin(selected_series)]

if "Model" in df.columns:
    Model_options = sorted(df["Model"].dropna().unique())
    selected_Model = st.sidebar.multiselect(
        "Select GPU Model",
        options=Model_options,
        default=Model_options
    )
    df = df[df["Model"].isin(selected_Model)]

# NumPy calculations safely
total_gpus = len(df)
if total_gpus > 0: #If there are gpu then it will calculate the average fps and tflops
    avg_fps = np.round(np.mean(df["Gaming_4K_FPS"]), 1) #calculating the average fps
    avg_tflops = np.round(np.mean(df["Compute_TFLOPS"]), 1) #calculating the average tflops
else: #If there are no gpu then it will set the average fps and tflops to 0
    avg_fps = 0
    avg_tflops = 0

# Summary metrics
st.subheader("GPU Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total GPUs Displayed", total_gpus)
col2.metric("Average 4K Gaming FPS", f"{avg_fps} FPS")
col3.metric("Average Compute Power", f"{avg_tflops} TFLOPS")

# Display a sample of the data
st.subheader("Dataset Preview")
st.write("The choosen GPU models and series are below.")
st.dataframe(df.head(12), use_container_width=True)

# Charts
st.subheader("Visualizations")

chart1, chart2 = st.columns(2)

with chart1:
    st.write("### 4K Gaming FPS by GPU Model")
    fig, ax = plt.subplots(figsize=(6, 4))
    sb.barplot(data=df, x="Model", y="Gaming_4K_FPS", hue="Series", ax=ax, palette="Set2")
    ax.set_xlabel("GPU Model")
    ax.set_ylabel("4K FPS")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    st.write("### Compute Performance (TFLOPS)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sb.barplot(data=df, x="Model", y="Compute_TFLOPS", hue="Series", ax=ax, palette="Set1")
    ax.set_xlabel("GPU Model")
    ax.set_ylabel("TFLOPS")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

chart3, chart4 = st.columns(2)

with chart3:
    st.write("### Power Consumption (TDP in Watts)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sb.barplot(data=df, x="Model", y="TDP(Watts)", hue="Series", ax=ax, palette="Dark2")
    ax.set_xlabel("GPU Model")
    ax.set_ylabel("TDP (Watts)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with chart4:
    st.write("### Distribution of VRAM Sizes")
    vram_data = df["VRAM_GB"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sb.histplot(data=vram_data, bins=10, kde=True, ax=ax, color="purple")
    ax.set_xlabel("VRAM Capacity (GB)")
    ax.set_ylabel("Number of GPUs")
    st.pyplot(fig)
    plt.close(fig)

# Compute average performance by series
st.subheader("Average Compute Power by Series")
if "Series" in df.columns:
    avg_perf = df.groupby("Series")[["Compute_TFLOPS", "Gaming_4K_FPS"]].mean().round(2)
    st.dataframe(avg_perf.reset_index(), use_container_width=True)

# Missing values check
with st.expander("Check missing values"):
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, use_container_width=True)

st.caption("This project is created by Priyanshu and all the records are tested and verified. The sources are given below.")
st.markdown("https://www.nvidia.com/en-in/geforce/graphics-cards/compare/")
st.markdown("https://nanoreview.net/")