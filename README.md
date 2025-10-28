# FPL Forecaster
This project uses historical Fantasy Premier League (FPL) data to predict the fantasy points a player might score in the next gameweek. The system downloads data from a GitHub repository, processes and enriches it with additional features (like fixture difficulty), trains an LSTM or GRU model, and finally produces predictions and an interactive Streamlit dashboard.

**Overview**
The pipeline consists of four main stages:

1.  **Data Ingestion**: Downloads key CSV files (teams, fixtures, player lists, player gameweek data, etc.) from a public GitHub repository and saves them locally.
2.  **Data Processing**: Loads the local data, computes additional features such as dynamic fixture difficulty and rolling averages (form). It also normalizes the features and converts the data into time-series sequences for the model.
3.  **Model Training and Prediction**: Builds and trains a recurrent neural network (LSTM or GRU) using a proper time-series validation strategy. The model learns from the historical sequences and predicts fantasy points for the next gameweek.
4.  **Visualization and Analysis**: Produces an interactive Streamlit dashboard showing the top 100 players by predicted fantasy points versus their price, and also provides a ranked list of players by position.

**Dashboard Features**
*   **Data Management**: The dashboard now indicates if data exists locally and provides options to force a re-download or check for updates.
*   **Cached Model and Predictions**: The trained model and predictions are saved to disk to avoid re-running the pipeline on every startup. The application now handles corrupted prediction files gracefully.
*   **Overview Tab**: Visualizes the top 100 players' predicted points against their price.
*   **Predictions Tab**: Displays a list of players filterable by team and price for each position, including team names. The tab now remains active after filtering.
*   **Team Builder Tab**: Allows you to build a "dream team" based on a selected budget and formation, including team names.

**Directory Structure**
```
.
├── src/
│   ├── data_ingestion.py     # Downloads data from GitHub and saves it locally.
│   ├── data_processing.py    # Loads local data, computes new features, normalizes data, and creates LSTM input sequences.
│   ├── model.py              # Defines, trains, and tunes the LSTM/GRU model; includes prediction functions.
│   └── main.py               # Runs the complete pipeline and Streamlit dashboard.
├── data/                     # Directory where the ingested CSV files are saved.
├── fpl_prediction.log        # Log file for the application.
├── requirements.txt          # Python dependencies.
└── README.md                 # This file.
```

**Requirements**
Install the required packages using the `requirements.txt` file:

```
pip install -r requirements.txt
```

**How to Run the Project**
1.  **Run the Streamlit Dashboard**:
    ```
    streamlit run src/main.py
    ```
2.  **Run the Full Pipeline**:
    - Open the Streamlit dashboard in your browser.
    - Click the "Run Full Pipeline" button to ingest data, process it, train the model, and generate predictions.

**Acknowledgements**
This project is based on the work from [bipan-sh/fpl-prediction-lstm](https://github.com/bipan-sh/fpl-prediction-lstm?tab=readme-ov-file) and uses data from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League/tree/master).
