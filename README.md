# Detecting Anomalies in Unlabeled Data Using Autoencoders

This repository provides an approach for detecting anomalies in unlabeled transactional data using Autoencoders. The primary objective is to identify businesses with abnormal transaction patterns, focusing on credit and debit transactions.

## Requirements

To run this project, you will need the following Python packages:

- pandas
- numpy
- seaborn
- matplotlib
- scipy
- scikit-learn
- keras

You can install the required libraries using the following command:

```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn keras
```

# File Structure
- `Data/challenge_txs.csv`: The transactional data used for anomaly detection. This should be placed in the `Data/` directory.
- `Anomaly Detection using Autoencoder.ipyb`: The main Python script for anomaly detection using Autoencoders.

## Description of the Code

### Data Preprocessing
- **Reading Data**: The `challenge_txs.csv` file is loaded into a Pandas DataFrame.
- **Data Cleaning**:
  - The column `a_idx` is dropped as it doesn't contribute to the analysis.
  - Missing values in the `counter_party_country` column are imputed with "Unknown."
- **Datetime Handling**: The `date` column is converted to datetime format, and new columns for `year`, `month`, `day`, `hour`, and `day_of_week` are extracted.

### Exploratory Data Analysis (EDA)
- Transaction type distribution is visualized (Credit vs. Debit).
- Transaction amounts are visualized using boxplots and histograms to identify outliers.
- Temporal patterns such as transaction counts by hour and day of the week are explored.

### Feature Engineering
- **Aggregating Data by Business**: Transaction data is aggregated at the `holding_party_id` level to create business-level features such as:
  - Total transactions
  - Total amount
  - Average transaction amount
  - Standard deviation of transaction amount
  - Unique counterparties
  - Credit/Debit totals
- **Credit-to-Debit Ratio**: A new feature is created to calculate the credit-to-debit ratio for each business.
- **Normalization**: The features are scaled using `StandardScaler` to prepare the data for model training.

### Model Building & Training
- An Autoencoder is built with Keras to detect anomalies at the business level. The autoencoder model is composed of an input layer, encoding layers, and decoding layers.
- The model is trained on the normalized business data using Mean Squared Error (MSE) as the loss function. The training process runs for 50 epochs with a batch size of 32.

### Anomaly Detection
- After training, reconstruction errors are calculated for each business by comparing the input features with the reconstructed features.
- A threshold for anomalies is set at the 95th percentile of the reconstruction error. Businesses with reconstruction errors above this threshold are labeled as anomalous.
- The results are visualized by plotting `total_credits` vs. `total_debits` for both normal and anomalous businesses.

## How to Use
1. Place the `challenge_txs.csv` file in the `Data/` directory.
2. Run the notebook `Anomaly Detection using Autoencoder.ipyb` to perform the anomaly detection process.
3. Review the visualizations and the output showing the number of anomalous businesses detected.

## Key Insights
- **Low Credit, High Debit**: This could indicate businesses spending more than they are earning, potentially pointing to cash flow issues.
- **High Credit, Low Debit**: This may suggest irregular business activities or a sudden change in transaction patterns.
- **Zero Credit and Zero Debit**: Businesses with no transactions may be inactive or dormant.

## Example Output
The script will output:
- Number of anomalous businesses detected.
- Visualizations showing the total credits vs. total debits for both normal and anomalous businesses.

## Contributions
Feel free to contribute to this project by forking the repository, making changes, and submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

