import argparse
import os
import re

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline


def is_key_period(normalized_period):
    """
    Determine if the PeriodID corresponds to a key period.
    """
    key_periods = [0, 45, 110, 120, 135,
                   160]  # Key moments: Kickoff, Halftime, End of 90 mins, Start of Extensions, Halftime in Extensions, End of Extensions
    margin = 5  # Allowable deviation

    # Check if normalized_period is close to any key period
    is_near_key_period = any(abs(normalized_period - kp) <= margin for kp in key_periods)

    # Check if it's within the halftime or extensions halftime ranges
    in_halftime_range = 45 <= normalized_period <= 65
    in_extensions_halftime_range = 135 <= normalized_period <= 145

    return int(is_near_key_period or in_halftime_range or in_extensions_halftime_range)


def extract_embeddings_glove_batch(df, device):
    """
    Batch process GloVe embeddings for all tweets in the DataFrame.
    """
    # Loading pre-trained GloVe embeddings
    glove_model = api.load("glove-twitter-200")  # 200-dimensional embeddings
    # Precomputing embeddings for all words in the vocabulary
    vocab = {word: torch.tensor(glove_model[word], device=device) for word in glove_model.key_to_index}

    def get_tweet_embedding(tweet):
        words = tweet.split()
        valid_embeddings = [vocab[word] for word in words if word in vocab]
        return (torch.stack(valid_embeddings).mean(dim=0)).cpu().numpy() if valid_embeddings else torch.zeros(
            glove_model.vector_size, device=device).cpu().numpy()

    # Applying embedding function to the DataFrame
    return df['Cleaned_Tweet_Glove'].apply(get_tweet_embedding)


def extract_embeddings_bert(df, device, batch_size=16):
    """
    Extract BERT embeddings using batch processing.
    """
    # Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    # Preparing batches
    tweets = df['Cleaned_Tweet'].tolist()
    embeddings = []

    # Processing in batches
    for i in tqdm(range(0, len(tweets), batch_size), desc="Extracting BERT Embeddings"):
        batch = tweets[i:i + batch_size]
        # Tokenizing batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        # Forward pass through model
        outputs = model(**inputs)
        # Mean pooling over the last hidden state, moving to CPU, and converting to NumPy
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.append(batch_embeddings)

    # Concatenating all batch embeddings
    return np.vstack(embeddings)


def remove_duplicates_retweets_and_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removing duplicate tweets, retweets, and tweets containing @mentions.
    """
    # Dropping exact duplicates
    df = df.drop_duplicates(subset=['Tweet'])

    tweet_strings = df['Tweet'].str
    # Removing retweets
    df = df[~(tweet_strings.startswith('RT') | tweet_strings.contains('@'))]

    return df


def process_single_tweet_context_aware(tweet):
    """Clean tweet for sentiment analysis and BERT embeddings while retaining context:
    - Remove URLs
    - Retain relevant punctuation and emojis
    - Retain stopwords
    - Keep the natural sentence structure
    """
    # Removing URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # Removing unwanted characters and retaining useful punctuation
    tweet = re.sub(r"[^\w\s!?:.,;#'\"-]", '',
                   tweet)  # Retaining !, ?, :, dots, commas, semicolons, hashtags, quotes and hyphens

    # Removing leading and trailing whitespaces
    tweet = tweet.strip()

    return tweet


class TweetPreprocessor:
    """
    A class for preprocessing tweets with configurable options.
    """

    def __init__(self, data_dir, device, mode):
        self.data_dir = data_dir
        self.device = device
        self.mode = mode

        # Loading stopwords but keeping relevant football terms
        self.stop_words = set(stopwords.words('english')) - {'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                                                             'again', 'goal', 'score', 'yellow', 'red', 'card',
                                                             'penalty', 'kick', 'foul'}

        self.tokenizer = TweetTokenizer(preserve_case=True, reduce_len=False)  # Preserving case for GloVe embeddings

        # Setting up sentiment analyzer
        model = "j-hartmann/emotion-english-distilroberta-base"  # Emotion classification model
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=model, device=self.device)

        # Regex patterns
        self.repeated_chars_pattern = re.compile(r'(\w)\1{2,}')  # Matches repeated characters (3 or more times)
        self.score_pattern = re.compile(r'\b\d+\s?[-:]\s?\d+\b')  # Matches scores like '3-1', '3 : 1', etc.
        common_columns = ['ID', 'PeriodID', 'Timestamp', 'Cleaned_Tweet', 'Sentiment_Score', 'Sentiment_anger',
                          'Sentiment_fear', 'Sentiment_joy', 'Sentiment_sadness', 'Sentiment_surprise',
                          'Exclamation_Count', 'Question_Count', 'Uppercase_Ratio', 'Repeated_Char_Word_Ratio',
                          'Is_Key_Period', 'Gives_Score', 'BERT_Embedding', 'GloVe_Embedding']
        if mode == 'train':
            self.columns_to_save = common_columns + ['EventType']
        else:
            self.columns_to_save = common_columns

    def process_single_tweet_glove(self, tweet):
        """Clean tweet for GloVe embeddings by:
        - Removing URLs
        - Retaining !, ?, :, dots, semicolons, hashtags, quotes and hyphens
        - Tokenizing
        - Removing stopwords
        """
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r"[^\w\s!?:.,;#'\"-]", '', tweet)
        tokens = self.tokenizer.tokenize(tweet.strip())
        return " ".join([token for token in tokens if token.lower() not in self.stop_words])

    def repeated_char_word_ratio(self, text):
        """Calculate the ratio of words with repeated characters to total words."""
        words = text.split()
        if not words:
            return 0
        repeated_count = sum(1 for word in words if self.repeated_chars_pattern.search(word))
        return repeated_count / len(words)

    def extract_features(self, df):
        """Extract features like sentiment, character repetition, punctuation counts, etc."""
        # Sentiment Analysis
        # Extracting both sentiment label and score
        # Using batch processing for efficiency
        sentiments = self.sentiment_analyzer(df['Cleaned_Tweet'].tolist(), batch_size=16)

        df['Sentiment_Label'] = [s['label'] for s in sentiments]
        df['Sentiment_Score'] = [s['score'] for s in sentiments]

        # Performing One-Hot Encoding on sentiment labels
        one_hot = pd.get_dummies(df['Sentiment_Label'], prefix='Sentiment')
        df = pd.concat([df, one_hot], axis=1)

        # Special Character Counts
        df['Exclamation_Count'] = df['Cleaned_Tweet'].str.count('!')
        df['Question_Count'] = df['Cleaned_Tweet'].str.count(r'\?')

        # Uppercase characters Ratio excluding white spaces
        df['Uppercase_Ratio'] = df['Cleaned_Tweet'].apply(
            lambda x: (sum(1 for c in x if c.isupper()) / sum(1 for c in x if not c.isspace())) if len(
                x.strip()) else 0)

        df['Repeated_Char_Word_Ratio'] = df['Cleaned_Tweet'].apply(self.repeated_char_word_ratio)

        # Key Period flag
        df['Is_Key_Period'] = df['PeriodID'].apply(
            lambda x: is_key_period(x - 5) if x - 5 >= 0 else 0)  # We subtract 5 to set kickoff near 0

        # Score Mention
        df['Gives_Score'] = df['Cleaned_Tweet'].apply(lambda x: 1 if self.score_pattern.search(x) else 0)

        return df

    def preprocess_and_extract_features(self):
        """Process all files in the provided data directory and extract features."""
        all_data = []
        for file in tqdm(os.listdir(self.data_dir)):
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)

            df.drop('MatchID', axis=1, inplace=True)

            # Removing duplicates, retweets, and tweets with mentions
            df = remove_duplicates_retweets_and_mentions(df)

            df['Cleaned_Tweet'] = df['Tweet'].apply(process_single_tweet_context_aware)
            df['Cleaned_Tweet_Glove'] = df['Tweet'].apply(self.process_single_tweet_glove)

            # Replacing NaN values with an empty string
            df['Cleaned_Tweet'] = df['Cleaned_Tweet'].fillna("").astype(str)
            df['Cleaned_Tweet_Glove'] = df['Cleaned_Tweet_Glove'].fillna("").astype(str)

            # Feature Extraction
            df = self.extract_features(df)

            all_data.append(df)

        # Combining all preprocessed data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Saving combined_data before computing embeddings
        combined_data.to_csv(f"backup_data/{self.mode}_combined_data_pre_embeddings.csv", index=False)

        # Extracting GloVe Embeddings
        combined_data['GloVe_Embedding'] = extract_embeddings_glove_batch(combined_data, self.device)

        # Extracting BERT Embeddings
        bert_embeddings = extract_embeddings_bert(combined_data, self.device)

        # Saving as a .npy file
        np.save(f"backup_data/{self.mode}_bert_embeddings.npy", bert_embeddings)

        bert_embeddings_list = [embedding for embedding in bert_embeddings]

        combined_data['BERT_Embedding'] = bert_embeddings_list

        return combined_data[self.columns_to_save]


def main():
    parser = argparse.ArgumentParser(description="Preprocess tweets and extract features for sub-event detection.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the CSV files to process.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for processed data.')
    parser.add_argument('--device', type=str, default='cpu', help='Device for hardware acceleration (default: cpu)')
    parser.add_argument('--mode', type=str, default='train', help='Mode for preprocessing (default: train)')
    args = parser.parse_args()

    print("Initializing Tweet Preprocessor...")
    preprocessor = TweetPreprocessor(args.data_dir, args.device, args.mode)

    print("Preprocessing and extracting features...")
    processed_data = preprocessor.preprocess_and_extract_features()

    print(f"Saving processed data to {args.output_file}...")
    processed_data.to_csv(args.output_file, index=False)
    print("Processing completed.")


if __name__ == "__main__":
    main()
