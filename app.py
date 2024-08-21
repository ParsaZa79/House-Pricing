import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import your existing functions here
from data_analysis import load_and_preprocess_data, build_and_evaluate_model, tune_hyperparameters

def main():
    st.set_page_config(page_title="House Price Analysis", layout="wide")
    st.title("House Price Analysis Dashboard")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data(uploaded_file)

        # Sidebar for navigation
        st.sidebar.title("Navigation")
        pages = ["Data Overview", "Exploratory Data Analysis", "Model Performance", "Feature Importance"]
        selection = st.sidebar.radio("Go to", pages)

        if selection == "Data Overview":
            st.header("Data Overview")
            st.write(X.head())
            st.write(f"Number of features: {X.shape[1]}")
            st.write(f"Number of samples: {X.shape[0]}")

        elif selection == "Exploratory Data Analysis":
            st.header("Exploratory Data Analysis")
            
            # Feature distributions
            st.subheader("Feature Distributions")
            fig = make_subplots(rows=3, cols=3, subplot_titles=X.columns)
            for i, col in enumerate(X.columns):
                row = i // 3 + 1
                col_num = i % 3 + 1
                fig.add_trace(go.Histogram(x=X[col], name=col), row=row, col=col_num)
            fig.update_layout(height=800, width=1000, title_text="Feature Distributions")
            st.plotly_chart(fig)

            # Correlation heatmap
            st.subheader("Feature Correlations")
            corr = X.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            fig.update_layout(height=800, width=1000, title_text="Correlation Heatmap")
            st.plotly_chart(fig)

            # Boxplots
            st.subheader("Feature Boxplots")
            fig = go.Figure()
            for col in X.columns:
                fig.add_trace(go.Box(y=X[col], name=col))
            fig.update_layout(height=600, width=1000, title_text="Feature Boxplots")
            st.plotly_chart(fig)

        elif selection == "Model Performance":
            st.header("Model Performance")
            model = build_and_evaluate_model(X_train, X_test, y_train, y_test)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("Mean Absolute Error", f"{mae:.2f}")
            col3.metric("R-squared Score", f"{r2:.2f}")

            # Cross-validation results
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            st.write("Cross-validation MSE scores:", -cv_scores)
            st.write(f"Average CV MSE: {-cv_scores.mean():.2f}")

            # Scatter plot of predicted vs actual values
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                             title='Predicted vs Actual Values')
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Ideal Prediction'))
            st.plotly_chart(fig)

        elif selection == "Feature Importance":
            st.header("Feature Importance")
            best_model = tune_hyperparameters(X_train, y_train)
            
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
            feature_importance = feature_importance.sort_values('importance', ascending=False)

            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Feature Importance', labels={'importance': 'Importance', 'feature': 'Feature'})
            fig.update_layout(height=600, width=1000)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()