import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go
from streamlit.components.v1 import html
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

st.set_page_config(
    page_title="Fraud detection",
    page_icon="💻",
)


st.write('# 💻 Stock market analysis: Fraud detection')

st.markdown(
    """
Stock market prediction can certainly be used for fraud detection, as analyzing patterns in stock market data can help identify potential instances of market manipulation or insider trading. This is an important use case as it helps protect and inform investors, ensures the stability of financial systems, and prevents systemic risk and competition between financial centers.

Fraud detection in stock markets requires advanced analytical techniques that can identify patterns indicating market manipulation or insider trading. The goal of fraud detection is to prevent illicit activities that undermine the integrity of the stock market and cause financial losses for investors.

To detect fraud in the stock market, there is a need to analyze large volumes of financial data in real-time. The analysis should be able to detect any unusual or abnormal patterns in trading activity, such as sudden surges in trading volumes, abnormal price fluctuations, or changes in trading patterns. The system should also be able to flag any suspicious activity for further investigation by regulatory authorities.

"""
)


st.write("## 🔍 Upload your file or select a stock on the sidebar!")

# Define a function to perform anomaly detection using a given algorithm
def detect_anomalies(df, algorithm):
    # Train the anomaly detection model
    algorithm.fit(df)

    # Predict the anomaly score for each data point
    anomaly_scores = algorithm.decision_function(df)

    # Convert the anomaly scores to a probability distribution
    min_anomaly_score = min(anomaly_scores)
    max_anomaly_score = max(anomaly_scores)
    anomaly_probabilities = (anomaly_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # Return the anomaly scores and probabilities
    return anomaly_scores, anomaly_probabilities

# Define a function for anomaly detection
def detect_anomalies2(df, algorithm):
    # Train the anomaly detection model
    algorithm.fit(df)

    # Predict the anomaly score for each data point
    anomaly_scores = algorithm.negative_outlier_factor_

    # Convert the anomaly scores to a probability distribution
    min_anomaly_score = min(anomaly_scores)
    max_anomaly_score = max(anomaly_scores)
    anomaly_probabilities = (anomaly_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # Return the anomaly scores and probabilities
    return anomaly_scores, anomaly_probabilities


# Define the Streamlit app
def app():

    # Allow the user to upload a CSV file or select from a dropdown list
    st.write("**👈 Click on the sidebar to chose a specific stock!** ")

    file = st.file_uploader("Upload a CSV file:", type=["csv"])
    if not file:
        # options = ["stock_market_data_TSLA.csv", "stock_market_data_AMZN.csv", "stock_market_data_FB.csv"]
        # filename = st.selectbox("Select a CSV file", options)
        # file = f"./data_alpha/{filename}"
        # Define a dictionary of CSV file names and paths
        csv_files = {
            'AAL: American Airlines Group Inc.': './data_alpha/stock_market_data_AAL.csv',
            'AAPL: Apple Inc.': './data_alpha/stock_market_data_AAPL.csv',
            'AIR: Airbus Group': './data_alpha/stock_market_data_AIR.csv',
            'AMZN: Amazon.com, Inc.': './data_alpha/stock_market_data_AMZN.csv',
            'BAC: Bank of America Corporation': './data_alpha/stock_market_data_BAC.csv',
            'CAPP: Capgemini': './data_alpha/stock_market_data_CAPP.csv',
            'COST: Costco Wholesale Corp.': './data_alpha/stock_market_data_COST.csv',
            'MC.PA: LVMH Moët Hennessy': './data_alpha/stock_market_data_MC.PA.csv',
            'META: Meta Platforms, Inc.	': './data_alpha/stock_market_data_META.csv',
            'MSFT: 	Microsoft Corporation': './data_alpha/stock_market_data_MSFT.csv',
            'NFLX: Netflix, Inc.': './data_alpha/stock_market_data_NFLX.csv',
            'NVDA: NVIDIA Corporation': './data_alpha/stock_market_data_NVDA.csv',
            'RIOT: Riot Platforms, Inc.': './data_alpha/stock_market_data_RIOT.csv',
            'SAN.PA: Sanofi': './data_alpha/stock_market_data_SAN.PA.csv',
            'TSLA: Tesla, Inc.': './data_alpha/stock_market_data_TSLA.csv',
            'TTE: TotalEnergies': './data_alpha/stock_market_data_TTE.csv'
        }

        # Display a dropdown menu for the user to select a CSV file
        csv_file = st.sidebar.selectbox("Select a stock:", list(csv_files.keys()))
        file = f"{csv_files[csv_file]}"


    # Load the data from the CSV file
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Perform anomaly detection using the IsolationForest algorithm
    algorithm = IsolationForest(n_estimators=100)
    anomaly_scores_if, anomaly_probabilities_if = detect_anomalies(df, algorithm)

    # Perform anomaly detection using the LocalOutlierFactor algorithm
    algorithm = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    anomaly_scores_lof, anomaly_probabilities_lof = detect_anomalies2(df, algorithm)

    st.write("""
    **This fraud detection application uses two primary techniques to identify potential fraudulent activity:
    isolation forest and local outlier factor.**
    """)
    st.write("## 📊 Visualisations")

    # Visualize the anomaly scores and probabilities
    st.subheader("Anomaly Scores")
    st.line_chart(anomaly_scores_if)

    st.subheader("Anomaly Probabilities")
    st.line_chart(anomaly_probabilities_if)
    st.write("These linecharts shows the stock's anomaly probalities and scores over time. The anomaly probability represents the degree of abnormality of a data point, with 0 indicating that the point is very likely to be a normal point and 1 indicating that the point is very likely to be an anomalous point. Therefore, in the scatter plot, the points with a higher anomaly probability (closer to 1) are more likely to be anomalous than the points with a lower anomaly probability (closer to 0).")

    # Visualize the anomaly scores using a scatter plot
    fig = px.scatter(df, x=df.index, y='4. close', color=anomaly_scores_lof, color_continuous_scale='portland_r')
    fig.update_layout(title='Anomalies? (closing price)', xaxis_title='Date', yaxis_title='Score')
    st.plotly_chart(fig)

    fig = px.scatter(df, x=df.index, y='1. open', color=anomaly_scores_lof, color_continuous_scale='portland_r')
    fig.update_layout(title='Anomalies? (opening price)', xaxis_title='Date', yaxis_title='Score')
    st.plotly_chart(fig)

    st.write("This scatter plot shows the stock's closing and opening price over time, with the anomaly scores represented by color. The darker the color, the more anomalous the data point is. Anomalous data points may indicate potential fraud, errors in data recording, or other abnormal events that could affect the stock's value. You can use this information to investigate the causes of the anomalies and make informed decisions about trading or investment strategies.")

    fig = px.scatter(df, x=df.index, y='6. volume', color=anomaly_scores_lof, color_continuous_scale='blackbody')
    fig.update_layout(title="Anomaly Scores", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig)

    st.write("This scatter plot of the volume of a stock over time. The color of each point on the scatter plot represents the anomaly score, with blue indicating a lower anomaly score and yellow indicating a higher anomaly score. This plot can be useful for detecting anomalies in the volume of the financial instrument over time, which could be indicative of potential fraud or other irregularities.")
    

    st.write("# 🔍 For a specific year")

    anomaly_threshold = 0.5
    year_range = df.index.year.unique().tolist()
    selected_year = st.slider('Select a year:', min_value=min(year_range), max_value=max(year_range))

    # Filter data for selected year
    df_selected_year = df[df.index.year == selected_year]

   # Define threshold for anomaly detection
    anomaly_threshold = 0.5

    # Filter data for selected year
    df_selected_year = df[df.index.year == selected_year]

    # Calculate anomaly scores using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    anomaly_scores_lof = lof.fit_predict(df_selected_year[['4. close', '6. volume']])

    # Create scatter plot of closing prices with anomaly scores
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_selected_year.index, y=df_selected_year['4. close'], mode='markers',
                            marker=dict(color=anomaly_scores_lof, colorscale='orrd_r', opacity=0.5)))
    fig.update_layout(title=f"Closing Prices and Anomaly Scores for {selected_year}",
                    xaxis_title="Date", yaxis_title="Closing Price",
                    coloraxis=dict(colorbar=dict(title="Anomaly Score")),
                    height=500)
    st.plotly_chart(fig)

    # Calculate number of potential anomalies
    num_anomalies = sum(1 for score in anomaly_scores_lof if score < -anomaly_threshold)

    # Display KPI box
    st.markdown(f"""
        <div style='background-color:rgba(0, 0, 0, 0.0); border:2px solid #ff4b4b; padding:5px, margin-bottom:10px'>
            <h2 style='text-align:center'>Potential anomalies</h2>
            <h3 style='text-align:center'>{num_anomalies}</h3>
        </div>
    """, unsafe_allow_html=True)

    dfcopy = df.copy()
    
    # Calculate the average volume per transaction (assumes an average transaction size of 100 shares)
    avg_vol_per_transaction = 100

    # Calculate the total number of transactions per day
    dfcopy['num_transactions'] = dfcopy['6. volume'] / avg_vol_per_transaction

    df_selected_yearcopy = dfcopy[dfcopy.index.year == selected_year]

    # # Calculate the total number of transactions for the selected year
    total_transactions = df_selected_yearcopy['num_transactions'].sum()

    # Calculate total revenue
    total_revenue = df['4. close'].sum()

    # Calculate total amount lost due to fraud
    anomaly_mask = anomaly_scores_lof == -1
    total_fraud_loss = df_selected_year.loc[anomaly_mask, '4. close'].sum()

    # Calculate Percentage of fraud losses as a proportion of total revenue
    pct_fraud_loss = (total_fraud_loss / total_revenue) * 100

    # Calculate the percentage of transactions that are flagged as potentially fraudulent
    pct_fraudulent_transactions = len(df_selected_year.loc[anomaly_mask, '4. close']) / len(df) * 100


    # Display the KPIs
    st.markdown(f"""
    <div style='background-color:rgba(0, 0, 0, 0.0); border:2px solid #ff4b4b; padding:5px, margin-bottom:10px'>
        <h2 style='text-align:center'>{selected_year} KPIs</h2>
        <div style='display:flex; justify-content:space-around'>
            <div>
                <p style='text-align:center'>Number of Transactions</p>
                <h3 style='text-align:center'>{total_transactions:.0f}</h3>
            </div>
            <div>
                <p style='text-align:center'>Total Revenue</p>
                <h3 style='text-align:center'>{total_revenue:.0f}</h3>
            </div>
        </div>
        <div style='display:flex; justify-content:space-around'>
            <div>
                <p style='text-align:center'>Transactions Potentially Fraudulent</p>
                <h3 style='text-align:center'>{pct_fraudulent_transactions:.2f}%</h3>
            </div>
            <div>
                <p style='text-align:center'>Fraud Losses (% of Total Revenue)</p>
                <h3 style='text-align:center'>{pct_fraud_loss:.2f}%</h3>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
                    

    

if __name__ == "__main__":
    app()
    # Create empty space
    space = st.empty()

    # Add vertical space
    empty_line = st.empty()
    empty_line.markdown("")
    
    st.write("""
    Please note that these techniques are not foolproof and may occasionally flag legitimate transactions as potentially
    fraudulent. As such, our application should be used as a tool to aid in the identification of potential fraud,
    rather than a definitive indicator of fraudulent activity.
    """)

    st.write("# Now, it's up to you! 🫵")
