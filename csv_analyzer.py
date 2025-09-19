import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype

# App Configuration
st.set_page_config(page_title="Advanced CSV Data Analyzer", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1 {
            color: #2e86de;
        }
        .st-bq {
            border-left: 4px solid #2e86de;
            padding-left: 1rem;
        }
        .stButton>button {
            background-color: #2e86de;
            color: white;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2e86de;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Advanced CSV Data Analyzer")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # Tabs
    tabs = st.tabs(["🔍 Overview", "🧼 Clean Data",
                   "📈 Analysis", "📊 Visuals", "🤖 PCA/ML"])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("🔍 Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("🧾 Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Rows:**", df.shape[0])
                st.write("**Columns:**", df.shape[1])
                st.write("**Column Names:**", list(df.columns))
            with col2:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes.reset_index().rename(
                    columns={"index": "Column", 0: "Type"}), height=300)

        with st.expander("📈 Descriptive Statistics"):
            selected = st.multiselect(
                "Select columns for statistics", all_cols, key="desc_cols")
            if selected:
                st.dataframe(df[selected].describe(include='all'), height=400)

        with st.expander("⚠️ Missing Values"):
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                st.dataframe(missing)
            else:
                st.success("No missing values detected!")

        st.download_button("⬇️ Download Current Dataset", df.to_csv(index=False),
                           "cleaned_data.csv", "text/csv")

    # --- Clean Data Tab ---
    with tabs[1]:
        st.subheader("🧼 Clean & Transform Data")

        with st.expander("📍 Drop Columns"):
            cols_to_drop = st.multiselect(
                "Select columns to drop", all_cols, key="drop_cols")
            if st.button("Drop Selected Columns"):
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"Dropped columns: {cols_to_drop}")
                    st.session_state.df = df
                    st.rerun()()

        with st.expander("💡 Fill Missing Values"):
            col = st.selectbox("Select a column", all_cols, key="fill_col")
            method = st.selectbox(
                "Fill method", ["Mean", "Median", "Mode", "Custom Value"], key="fill_method")

            if method == "Custom Value":
                custom_value = st.text_input(
                    "Enter custom value", key="custom_fill")

            if st.button("Fill Missing"):
                if method == "Mean" and is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median" and is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Mode":
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        st.warning(f"No mode available to fill {col}.")
                elif method == "Custom Value" and custom_value:
                    try:
                        # Try to convert to float if column is numeric
                        if is_numeric_dtype(df[col]):
                            custom_value = float(custom_value)
                        df[col].fillna(custom_value, inplace=True)
                    except ValueError:
                        df[col].fillna(custom_value, inplace=True)

                st.success(f"Filled missing values in {col} using {method}.")
                st.session_state.df = df
                st.rerun()()

        with st.expander("🔢 Convert Data Type"):
            col = st.selectbox("Column to convert",
                               all_cols, key="convert_col")
            dtype = st.selectbox(
                "New type", ["int", "float", "str", "category"], key="convert_type")
            if st.button("Convert Type"):
                try:
                    df[col] = df[col].astype(dtype)
                    st.success(f"Converted {col} to {dtype}")
                    st.session_state.df = df
                    st.rerun()()
                except Exception as e:
                    st.error(f"Error: {e}")

        with st.expander("🔍 Filter Rows"):
            filter_col = st.selectbox(
                "Select column to filter", all_cols, key="filter_col")
            if is_numeric_dtype(df[filter_col]):
                min_val, max_val = float(
                    df[filter_col].min()), float(df[filter_col].max())
                selected_range = st.slider(
                    "Select range", min_val, max_val, (min_val, max_val), key="num_filter")
                if st.button("Apply Numeric Filter"):
                    df = df[(df[filter_col] >= selected_range[0]) &
                            (df[filter_col] <= selected_range[1])]
                    st.session_state.df = df
                    st.success(f"Filter applied! Rows remaining: {len(df)}")
                    st.rerun()()
            else:
                unique_values = df[filter_col].unique()
                selected_values = st.multiselect(
                    "Select values to keep", unique_values, default=unique_values, key="cat_filter")
                if st.button("Apply Categorical Filter"):
                    df = df[df[filter_col].isin(selected_values)]
                    st.session_state.df = df
                    st.success(f"Filter applied! Rows remaining: {len(df)}")
                    st.rerun()()

        if st.button("🔄 Reset to Original Data"):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Data reset to original state!")
            st.rerun()()

    # --- Analysis Tab ---
    with tabs[2]:
        st.subheader("📈 Exploratory Data Analysis")

        with st.expander("📉 Value Counts"):
            col = st.selectbox("Select column", all_cols,
                               key="value_counts_col")
            st.write(df[col].value_counts())

        with st.expander("📊 Correlation Matrix"):
            if len(numeric_cols) > 1:
                corr_method = st.selectbox("Correlation method", [
                                           "pearson", "kendall", "spearman"], key="corr_method")
                corr = df[numeric_cols].corr(method=corr_method)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f",
                            cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)

                if st.checkbox("Show Correlation Pairs"):
                    corr_pairs = corr.unstack().sort_values(ascending=False)
                    # Remove self-correlations
                    corr_pairs = corr_pairs[corr_pairs != 1]
                    st.write(corr_pairs.head(10))
            else:
                st.warning(
                    "Need at least 2 numeric columns for correlation analysis")

        with st.expander("📋 Column Statistics"):
            col = st.selectbox("Select a column", all_cols, key="col_stats")
            if is_numeric_dtype(df[col]):
                st.write(df[col].describe())

                tab1, tab2, tab3 = st.tabs(
                    ["Histogram", "Box Plot", "Violin Plot"])
                with tab1:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    st.pyplot(fig)
                with tab2:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    st.pyplot(fig)
                with tab3:
                    fig, ax = plt.subplots()
                    sns.violinplot(x=df[col], ax=ax)
                    st.pyplot(fig)
            else:
                tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[col].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                with tab2:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    df[col].value_counts().plot(
                        kind='pie', autopct='%1.1f%%', ax=ax)
                    st.pyplot(fig)

    # --- Visuals Tab ---
    with tabs[3]:
        st.subheader("📊 Visualization")

        # New 3D Visualization Section
        with st.expander("🌐 3D Visualizations", expanded=True):
            st.subheader("3D Scatter Plot")
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, key="3d_x")
                with col2:
                    y_axis = st.selectbox("Y-axis", numeric_cols, key="3d_y")
                with col3:
                    z_axis = st.selectbox("Z-axis", numeric_cols, key="3d_z")

                color_col = st.selectbox(
                    "Color by", [None] + all_cols, key="3d_color")
                size_col = st.selectbox(
                    "Size by", [None] + numeric_cols, key="3d_size")

                fig = px.scatter_3d(
                    df,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=color_col if color_col else None,
                    size=size_col if size_col else None,
                    hover_name=df.index,
                    title=f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Need at least 3 numeric columns for 3D scatter plot")

        with st.expander("📍 Plot Settings"):
            plot_type = st.selectbox("Select Plot Type",
                                     ["Histogram", "Box Plot", "Scatter Plot",
                                         "Line Plot", "Bar Chart"],
                                     key="plot_type")

            if plot_type == "Histogram":
                col = st.selectbox("Column", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 5, 100, 20, key="hist_bins")
                fig = px.histogram(df, x=col, nbins=bins)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Box Plot":
                col = st.selectbox("Column", numeric_cols, key="box_col")
                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatter Plot":
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color by (optional)", [
                                         None] + all_cols, key="scatter_color")
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Line Plot":
                x_col = st.selectbox("X-axis", all_cols, key="line_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
                fig = px.line(df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Bar Chart":
                x_col = st.selectbox("X-axis", all_cols, key="bar_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="bar_y")
                fig = px.bar(df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

    # --- PCA & ML Tab ---
    with tabs[4]:
        st.subheader("🤖 PCA - Dimensionality Reduction")

        if len(numeric_cols) >= 2:
            scale = st.checkbox(
                "🔵 Standardize Data (Recommended)", value=True, key="pca_scale")
            if scale:
                st.caption(
                    "Standardization centers and scales numeric columns for better PCA performance.")

            n_components = st.slider("Number of Components", 2, min(
                len(numeric_cols), 10), 2, key="pca_ncomp")

            data = df[numeric_cols].dropna()
            if scale:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
            else:
                data_scaled = data

            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            pca_df = pd.DataFrame(pca_result, columns=[
                                  f'PC{i+1}' for i in range(n_components)])

            st.write("Explained Variance Ratio:")
            st.bar_chart(pca.explained_variance_ratio_)

            st.subheader("PCA Visualization")

            # 2D/3D PCA Visualization Selection
            pca_dim = st.radio("Visualization Dimension", [
                               "2D", "3D"], horizontal=True, key="pca_dim")

            if pca_dim == "2D":
                color_col = st.selectbox(
                    "Color points by", [None] + all_cols, key="pca_color")
                fig = px.scatter(pca_df, x='PC1', y='PC2', color=df[color_col] if color_col else None,
                                 title="2D PCA Plot")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if n_components >= 3:
                    color_col = st.selectbox(
                        "Color points by", [None] + all_cols, key="pca_color_3d")
                    fig = px.scatter_3d(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color=df[color_col] if color_col else None,
                        title="3D PCA Plot"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Need at least 3 components for 3D visualization")

            st.write("PCA Component Loadings:")
            loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],
                                    index=numeric_cols)
            st.dataframe(loadings)
        else:
            st.warning("Need at least 2 numeric columns for PCA.")

else:
    st.info("📁 Please upload a CSV file to get started.")
