import pandas as pd
from textblob import TextBlob

# Load CSV file
df = pd.read_csv('C:/Users/Lenovo/Desktop/task4/twitter_training.csv')

# Clean the text
df["Cleaned_Text"] = df["Text"].apply(lambda x: str(x).lower())

# Apply sentiment analysis
df["TextBlob_Score"] = df["Cleaned_Text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["TextBlob_Sentiment"] = df["TextBlob_Score"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

# Print results
print(df.head())
