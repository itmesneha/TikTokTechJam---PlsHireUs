import pandas as pd

# Read the CSV file
df = pd.read_csv('../datasets/processed_reviews_with_sentiment.csv')

# Print the head of the CSV file
print(df.head())