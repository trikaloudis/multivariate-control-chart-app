import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f, chi2, norm
import io

def load_data(uploaded_file):
    """Loads data from CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # openpyxl is required for this
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Multivariate Control Chart Dashboard (T² & PCA-Based)")
    st.markdown("""
    This application generates advanced multivariate control charts to monitor process stability.
    
    1.  **Upload your data** (CSV or Excel) in the sidebar.
    2.  The app assumes **Column A is 'Index'** and **Column B is 'Date'**. All other columns will be treated as process variables.
    3.  Adjust the **Significance Level (α)** and **PCA Variance Target** in the sidebar.
    """)

    # --- Sidebar ---
    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    
    alpha = st.sidebar.slider("Significance Level (α)", 0.001, 0.10, 0.05, 0.005,
                              help="The probability of a Type I error (false alarm). Common values are 0.05, 0.01, or 0.0027.")
    
    pca_var_target = st.sidebar.slider("PCA Explained Variance Target", 0.80, 1.0, 0.95, 0.01,
                                       help="The minimum cumulative variance the PCA model should explain. This drives automatic component selection.")

    if uploaded_file is None:
        st.info("Please upload a data file to begin analysis.")
        
        # Show a sample data structure
        st.subheader("Sample Data Format")
        st.markdown("Your data should look something like this. The 'Index' and 'Date' columns are used for plotting, and all other numeric columns are used for analysis.")
        sample_data = {
            'Index': [1, 2, 3, 4, 5],
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Param1': [10.2, 10.1, 9.9, 10.0, 10.3],
            'Param2': [4.5, 4.6, 4.4, 4.5, 4.7],
            'Param3': [1.1, 1.0, 1.2, 1.1, 1.0]
        }
        st.dataframe(pd.DataFrame(sample_data))
        return

    # --- Data Loading and Preprocessing ---
    df = load_data(uploaded_file)
    if df is None:
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Identify variable columns (all columns after Index and Date)
    if len(df.columns) < 3:
        st.error("Data must have at least 3 columns: Index, Date, and one variable.")
        return
        
    try:
        # Use column positions 0 and 1 as metadata
        meta_cols = df.columns[:2].tolist()
        var_cols = df.columns[2:].tolist()
        
        st.write(f"**Metadata columns:** {meta_cols[0]}, {meta_cols[1]}")
        st.write(f"**Variable columns for analysis:** {', '.join(var_cols)}")
        
        X = df[var_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Get corresponding index/date for plotting
        plot_index = X.index
        x_axis = df.loc[plot_index, meta_cols[1]]
        x_axis_label = meta_cols[1]
        
        # Fallback to index if Date column is not suitable
        try:
            # Convert x_axis to datetime objects. This is the fix.
            x_axis = pd.to_datetime(x_axis)
        except Exception:
            # If conversion fails, post a warning and fall back to the index column
            st.warning(f"Could not parse '{meta_cols[1]}' as dates. Falling back to '{meta_cols[0]}' for the x-axis.")
            x_axis = df.loc[plot_index, meta_cols[0]]
            x_axis_label = meta_cols[0]

    except Exception as e:
        st.error(f"Error processing columns. Ensure first two columns are Index/Date and the rest are numeric variables. Error: {e}")
        return

    if X.empty:
        st.error("No valid numeric data found in variable columns.")
        return

    n, p = X.shape
    if n <= p:
        st.error(f"Analysis failed: The number of observations ({n}) must be greater than the number of variables ({p}).")
        return

    # --- 1. Hotelling's T² Chart ---
    st.header("1. Hotelling's T² Control Chart")
    st.markdown("""
    This chart combines all variables into a single statistic to monitor the overall process. 
    It's sensitive to shifts in the mean of *any* variable or changes in their correlations.
    """)
    
    try:
        x_mean = X.mean()
        S = X.cov()
        S_inv = np.linalg.inv(S)
        
        diff = X - x_mean
        T2_values = diff.apply(lambda row: row.values @ S_inv @ row.values.T, axis=1)
        
        # Phase 1 UCL (F-distribution)
        UCL_T2 = ((p * (n - 1) * (n + 1)) / (n * (n - p))) * f.ppf(1 - alpha, p, n - p)

        fig_t2 = go.Figure()
        fig_t2.add_trace(go.Scatter(x=x_axis, y=T2_values, mode='lines+markers', name='T² Statistic',
                                    marker=dict(color='blue'), line=dict(color='blue')))
        
        # Add UCL
        fig_t2.add_shape(type='line', x0=x_axis.min(), y0=UCL_T2, x1=x_axis.max(), y1=UCL_T2,
                         line=dict(color='red', dash='dot', width=2), name=f'UCL (1-α={1-alpha:.3f})')
        
        # Highlight OOC points
        ooc_t2_indices = T2_values[T2_values > UCL_T2].index
        if not ooc_t2_indices.empty:
            fig_t2.add_trace(go.Scatter(x=x_axis.loc[ooc_t2_indices], y=T2_values.loc[ooc_t2_indices],
                                        mode='markers', name='Out of Control',
                                        marker=dict(color='red', size=10, symbol='x')))

        fig_t2.update_layout(
            title="Hotelling's T² Chart",
            xaxis_title=x_axis_label,
            yaxis_title="T² Statistic",
            hovermode="x unified"
        )
        st.plotly_chart(fig_t2, use_container_width=True)
        st.write(f"**Upper Control Limit (UCL):** {UCL_T2:.4f}")
        st.write(f"**Out-of-Control Points:** {len(ooc_t2_indices)} ({(len(ooc_t2_indices)/n*100):.2f}%)")

    except np.linalg.LinAlgError:
        st.error("Failed to compute Hotelling's T²: The covariance matrix is singular. This can happen if variables are perfectly correlated or n <= p.")
        return
    except Exception as e:
        st.error(f"An error occurred during T² calculation: {e}")
        return

    # --- 2. Model-Driven Multivariate Control Chart (MDMVCC) ---
    st.header("2. Model-Driven Multivariate Control Chart (PCA)")
    st.markdown("""
    This section uses Principal Component Analysis (PCA) to model the normal process variation. 
    It's more robust and provides deeper insights than the standard T² chart.
    
    - **PCA T² Chart:** Monitors the "model space" (the major sources of variation).
    - **DModX Chart:** Monitors the "residual space" (how far each point is from the model).
    """)

    # Scale data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-select components
    pca = PCA(n_components=pca_var_target)
    T = pca.fit_transform(X_scaled) # Scores
    P = pca.components_.T           # Loadings
    n_comps = pca.n_components_
    
    st.info(f"PCA model was built with **{n_comps}** components, explaining **{pca.explained_variance_ratio_.sum():.2%}** of the total variance.")

    # --- PCA T² Chart (Model Space) ---
    st.subheader("PCA T² Chart (Model Space)")
    
    # T² on scores
    if n <= n_comps:
        st.error(f"Analysis failed: The number of observations ({n}) must be greater than the number of selected PCA components ({n_comps}). Try increasing observations or reducing the variance target.")
        return
        
    cov_T = np.cov(T, rowvar=False)
    mean_T = T.mean(axis=0)
    
    try:
        cov_T_inv = np.linalg.inv(cov_T)
        diff_T = T - mean_T
        T2_pca_values = np.array([row @ cov_T_inv @ row.T for row in diff_T])
        
        # UCL for PCA T²
        UCL_T2_pca = ((n_comps * (n - 1) * (n + 1)) / (n * (n - n_comps))) * f.ppf(1 - alpha, n_comps, n - n_comps)

        fig_t2_pca = go.Figure()
        fig_t2_pca.add_trace(go.Scatter(x=x_axis, y=T2_pca_values, mode='lines+markers', name='PCA T² Statistic',
                                        marker=dict(color='green'), line=dict(color='green')))
        
        fig_t2_pca.add_shape(type='line', x0=x_axis.min(), y0=UCL_T2_pca, x1=x_axis.max(), y1=UCL_T2_pca,
                             line=dict(color='red', dash='dot', width=2), name=f'UCL (1-α={1-alpha:.3f})')
        
        ooc_t2_pca_indices = plot_index[T2_pca_values > UCL_T2_pca]
        if len(ooc_t2_pca_indices) > 0:
            fig_t2_pca.add_trace(go.Scatter(x=x_axis.loc[ooc_t2_pca_indices], y=T2_pca_values[T2_pca_values > UCL_T2_pca],
                                            mode='markers', name='Out of Control',
                                            marker=dict(color='red', size=10, symbol='x')))

        fig_t2_pca.update_layout(title="PCA T² Chart (Model Space)", xaxis_title=x_axis_label, yaxis_title="PCA T² Statistic")
        st.plotly_chart(fig_t2_pca, use_container_width=True)
        st.write(f"**Upper Control Limit (UCL):** {UCL_T2_pca:.4f}")
        st.write(f"**Out-of-Control Points:** {len(ooc_t2_pca_indices)} ({(len(ooc_t2_pca_indices)/n*100):.2f}%)")

    except np.linalg.LinAlgError:
        st.error("Failed to compute PCA T²: The score covariance matrix is singular. This is highly unusual.")
    except Exception as e:
        st.error(f"An error occurred during PCA T² calculation: {e}")

    # --- DModX Chart (Residual Space) ---
    st.subheader("DModX Chart (Residual Space)")
    
    # Calculate residuals (E) and DModX
    E = X_scaled - (T @ P.T)
    DModX_values = np.sum(E**2, axis=1)
    
    # UCL for DModX (using chi2 approximation)
    mean_dmodx = np.mean(DModX_values)
    var_dmodx = np.var(DModX_values)
    
    if mean_dmodx > 0:
        g = var_dmodx / (2 * mean_dmodx)
        h = 2 * mean_dmodx**2 / var_dmodx
        UCL_DModX = g * chi2.ppf(1 - alpha, h)
    else:
        UCL_DModX = 0

    fig_dmodx = go.Figure()
    fig_dmodx.add_trace(go.Scatter(x=x_axis, y=DModX_values, mode='lines+markers', name='DModX',
                                   marker=dict(color='purple'), line=dict(color='purple')))

    fig_dmodx.add_shape(type='line', x0=x_axis.min(), y0=UCL_DModX, x1=x_axis.max(), y1=UCL_DModX,
                         line=dict(color='red', dash='dot', width=2), name=f'UCL (1-α={1-alpha:.3f})')
    
    ooc_dmodx_indices = plot_index[DModX_values > UCL_DModX]
    if len(ooc_dmodx_indices) > 0:
        fig_dmodx.add_trace(go.Scatter(x=x_axis.loc[ooc_dmodx_indices], y=DModX_values[DModX_values > UCL_DModX],
                                        mode='markers', name='Out of Control',
                                        marker=dict(color='red', size=10, symbol='x')))

    fig_dmodx.update_layout(title="DModX Chart (Residual Space)", xaxis_title=x_axis_label, yaxis_title="DModX")
    st.plotly_chart(fig_dmodx, use_container_width=True)
    st.write(f"**Upper Control Limit (UCL):** {UCL_DModX:.4f}")
    st.write(f"**Out-of-Control Points:** {len(ooc_dmodx_indices)} ({(len(ooc_dmodx_indices)/n*100):.2f}%)")

    # --- 3. Contribution Plots ---
    st.header("3. Contribution Plots (Root Cause Analysis)")
    st.markdown("""
    When a point is out of control, these plots help identify *which variables* are responsible.
    Select a point from the dropdown to investigate.
    """)

    # Create a list of all points for selection
    point_labels = [f"Index {df.loc[i, meta_cols[0]]} ({df.loc[i, meta_cols[1]]})" for i in plot_index]
    selected_point_label = st.selectbox("Select a point to analyze:", options=point_labels)
    
    if selected_point_label:
        selected_plot_index = point_labels.index(selected_point_label)
        selected_df_index = plot_index[selected_plot_index]
        
        st.write(f"**Analyzing:** {selected_point_label}")

        col1, col2 = st.columns(2)

        # --- T² Contribution Plot ---
        with col1:
            st.subheader("Hotelling T² Contributions")
            diff_i = X.loc[selected_df_index] - x_mean
            
            # T2 contribution: c_j = (x_i - x_mean)_j * [S_inv * (x_i - x_mean)]_j
            # This shows how much each variable (in its scaled/correlated form) adds to the T2 value.
            cont_t2 = diff_i * (diff_i.values @ S_inv)
            cont_t2_df = pd.DataFrame({'Variable': X.columns, 'Contribution': cont_t2}).sort_values(by='Contribution', ascending=False)
            
            fig_cont_t2 = px.bar(cont_t2_df, x='Variable', y='Contribution', 
                                 title=f"T² Contributions for {selected_point_label}",
                                 color='Contribution', color_continuous_scale=px.colors.diverging.RdYlBu_r)
            fig_cont_t2.update_layout(xaxis_title=None)
            st.plotly_chart(fig_cont_t2, use_container_width=True)

        # --- DModX Contribution Plot ---
        with col2:
            st.subheader("DModX Contributions")
            # DModX contribution is just the squared residual for each variable
            cont_dmodx = E[selected_plot_index, :]**2
            cont_dmodx_df = pd.DataFrame({'Variable': X.columns, 'Contribution': cont_dmodx}).sort_values(by='Contribution', ascending=False)
            
            fig_cont_dmodx = px.bar(cont_dmodx_df, x='Variable', y='Contribution', 
                                    title=f"DModX Contributions for {selected_point_label}",
                                    color='Contribution', color_continuous_scale=px.colors.sequential.OrRd)
            fig_cont_dmodx.update_layout(xaxis_title=None)
            st.plotly_chart(fig_cont_dmodx, use_container_width=True)

if __name__ == "__main__":
    main()
