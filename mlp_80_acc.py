import gc

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import torch 
import torch.nn as nn


def string_to_array(embedding_str):
    """
    Converting string representation of embedding to numpy array.
    Handles newlines and scientific notation.
    """
    # Removing brackets and newlines
    cleaned = embedding_str.strip('[]').replace('\n', ' ')
    return np.array([float(x) for x in cleaned.split()])


def process_large_dataset(filepath, chunk_size=10000, columns_to_load=None, mode='train'):
    """
    Processing large training or test CSV files in chunks.

    Args:
        filepath (str): Path to the CSV file
        chunk_size (int): Number of rows to process in each chunk
        columns_to_load (list): Columns to load from the CSV file
    """

    # Getting embedding dimensions from first row
    first_chunk = next(pd.read_csv(filepath, nrows=1, usecols=['GloVe_Embedding', 'BERT_Embedding'], chunksize=1))
    glove_dim = len(string_to_array(first_chunk['GloVe_Embedding'].iloc[0]))
    bert_dim = len(string_to_array(first_chunk['BERT_Embedding'].iloc[0]))
    # Initializing aggregators
    running_stats = {
        'sums': {},  # Storing sums for mean calculations
        'total_counts': {},  # Storing counts for mean calculations
        'first_values': {}  # Storing first occurrences for period-wise constant features
    }

    # Reading and processing the CSV file in chunks
    print("Processing chunks...")
    chunks_iterator = pd.read_csv(filepath, chunksize=chunk_size, usecols=columns_to_load)

    for chunk in tqdm(chunks_iterator):

        # Converting embeddings strings to arrays before processing
        chunk['GloVe_Embedding'] = chunk['GloVe_Embedding'].apply(string_to_array)
        chunk['BERT_Embedding'] = chunk['BERT_Embedding']  .apply(string_to_array)

        # Processing each ID in the chunk
        for id_group, group in chunk.groupby(['ID', 'PeriodID']):
            # Initializing if this ID hasn't been seen before
            if id_group not in running_stats['sums']:
                running_stats['sums'][id_group] = {
                    'Sentiment_joy': 0,
                    'Sentiment_anger': 0,
                    'Sentiment_fear': 0,
                    'Sentiment_sadness': 0,
                    'Sentiment_surprise': 0,
                    'Sentiment_Score': 0,
                    'Exclamation_Count': 0,
                    'Question_Count': 0,
                    'Uppercase_Ratio': 0,
                    'Repeated_Char_Word_Ratio': 0,
                    'Gives_Score': 0,
                    'GloVe_Embedding': np.zeros(glove_dim),
                    'BERT_Embedding': np.zeros(bert_dim)
                }
                running_stats['first_values'][id_group] = {
                    'Is_Key_Period': group['Is_Key_Period'].iloc[0],
                    'EventType': group['EventType'].iloc[0] if mode == 'train' else None,
                    'PeriodID': group['PeriodID'].iloc[0],
                    'ID': group['ID'].iloc[0]
                }
                running_stats['total_counts'][id_group] = 0

            # Updating sums for mean calculations
            n = len(group)
            running_stats['total_counts'][id_group] += n

            # Updating sums for each metric
            running_stats['sums'][id_group]['Sentiment_joy'] += group['Sentiment_joy'].sum()
            running_stats['sums'][id_group]['Sentiment_anger'] += group['Sentiment_anger'].sum()
            running_stats['sums'][id_group]['Sentiment_fear'] += group['Sentiment_fear'].sum()
            running_stats['sums'][id_group]['Sentiment_sadness'] += group['Sentiment_sadness'].sum()
            running_stats['sums'][id_group]['Sentiment_surprise'] += group['Sentiment_surprise'].sum()
            running_stats['sums'][id_group]['Sentiment_Score'] += group['Sentiment_Score'].sum()
            running_stats['sums'][id_group]['Exclamation_Count'] += group['Exclamation_Count'].sum()
            running_stats['sums'][id_group]['Question_Count'] += group['Question_Count'].sum()
            running_stats['sums'][id_group]['Uppercase_Ratio'] += group['Uppercase_Ratio'].sum()
            running_stats['sums'][id_group]['Repeated_Char_Word_Ratio'] += group['Repeated_Char_Word_Ratio'].sum()
            running_stats['sums'][id_group]['Gives_Score'] += group['Gives_Score'].sum()

            # Updating embedding sums
            running_stats['sums'][id_group]['GloVe_Embedding'] += np.sum(np.vstack(group['GloVe_Embedding']), axis=0)
            running_stats['sums'][id_group]['BERT_Embedding'] += np.sum(np.vstack(group['BERT_Embedding']), axis=0)

        # Forcing garbage collection after each chunk
        gc.collect()

    # Computing final aggregated results
    print("Computing final aggregations...")
    result_data = []

    for id_group in running_stats['total_counts'].keys():
        count = running_stats['total_counts'][id_group]

        result_dict = {
            'ID': running_stats['first_values'][id_group]['ID'],
            'PeriodID': running_stats['first_values'][id_group]['PeriodID'],
            'Tweet_Count': count,
            'Is_Key_Period': running_stats['first_values'][id_group]['Is_Key_Period'],
            'EventType': running_stats['first_values'][id_group]['EventType'],
            'Sentiment_joy': running_stats['sums'][id_group]['Sentiment_joy'] / count,
            'Sentiment_anger': running_stats['sums'][id_group]['Sentiment_anger'] / count,
            'Sentiment_fear': running_stats['sums'][id_group]['Sentiment_fear'] / count,
            'Sentiment_sadness': running_stats['sums'][id_group]['Sentiment_sadness'] / count,
            'Sentiment_surprise': running_stats['sums'][id_group]['Sentiment_surprise'] / count,
            'Sentiment_Score': running_stats['sums'][id_group]['Sentiment_Score'] / count,
            'Exclamation_Count': running_stats['sums'][id_group]['Exclamation_Count'],
            'Question_Count': running_stats['sums'][id_group]['Question_Count'],
            'Uppercase_Ratio': running_stats['sums'][id_group]['Uppercase_Ratio'] / count,
            'Repeated_Char_Word_Ratio': running_stats['sums'][id_group]['Repeated_Char_Word_Ratio'] / count,
            'Gives_Score': running_stats['sums'][id_group]['Gives_Score'],
            'GloVe_Embedding': running_stats['sums'][id_group]['GloVe_Embedding'] / count,
            'BERT_Embedding': running_stats['sums'][id_group]['BERT_Embedding'] / count
        }

        result_data.append(result_dict)

    # Creating final DataFrame
    aggregated_df = pd.DataFrame(result_data)

    return aggregated_df


def expand_embeddings(aggregated_df):
    """
    Expanding embedding features into separate columns.
    """
    print("Expanding embeddings...")

    # Getting embedding dimensions
    bert_embedding_dim = aggregated_df['BERT_Embedding'].iloc[0].shape[0]
    glove_embedding_dim = aggregated_df['GloVe_Embedding'].iloc[0].shape[0]

    # Creating column names
    bert_columns = [f'BERT_{i}' for i in range(bert_embedding_dim)]
    glove_columns = [f'GloVe_{i}' for i in range(glove_embedding_dim)]

    # Converting embeddings to DataFrames efficiently
    bert_features = pd.DataFrame(
        np.stack(aggregated_df['BERT_Embedding'].values),
        columns=bert_columns
    )

    glove_features = pd.DataFrame(
        np.stack(aggregated_df['GloVe_Embedding'].values),
        columns=glove_columns
    )

    # Combining DataFrames
    expanded_df = pd.concat([
        aggregated_df.drop(columns=['BERT_Embedding', 'GloVe_Embedding']),
        bert_features,
        glove_features
    ], axis=1)

    return expanded_df, bert_columns, glove_columns


def prepare_train_and_test_data(expanded_df_train, expanded_df_test, bert_columns, glove_columns, n_components=0.96):
    """
    Preparing data for model training with dimensionality reduction and scaling.
    """
    print("Preparing data for training...")

    columns_to_drop = ['ID', 'Sentiment_fear', 'EventType'] + glove_columns
    X_test = expanded_df_test.drop(columns=columns_to_drop)

    X_train_full = expanded_df_train.drop(columns=columns_to_drop)
    y_train_full = expanded_df_train['EventType']

    # Performing train-validation split
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.3, stratify=y_train_full
    )

    # # Dimensionality Reduction for BERT Columns
    # print("Reducing dimensionality of BERT features...")
    # pca_train_val = PCA(n_components=n_components)
    # pca_train_full = PCA(n_components=n_components)

    # # Applying PCA to train, validation, and test sets
    # bert_train_val = X_train_val[bert_columns]
    # bert_train_full = X_train_full[bert_columns]
    # bert_test = X_test[bert_columns]

    # bert_train_val_reduced = pca_train_val.fit_transform(bert_train_val)
    # bert_train_full_reduced = pca_train_full.fit_transform(bert_train_full)
    # bert_val_reduced = pca_train_val.transform(X_val[bert_columns])
    # bert_test_reduced = pca_train_full.transform(bert_test)

    # # Replace original BERT columns with reduced features
    # bert_train_val_reduced_columns = [f'BERT_PCA_{i}' for i in range(len(pca_train_val.explained_variance_ratio_))]
    # bert_train_full_reduced_columns = [f'BERT_PCA_{i}' for i in range(len(pca_train_full.explained_variance_ratio_))]
    # bert_train_val_df = pd.DataFrame(bert_train_val_reduced, columns=bert_train_val_reduced_columns,
    #                                  index=X_train_val.index)
    # bert_train_full_df = pd.DataFrame(bert_train_full_reduced, columns=bert_train_full_reduced_columns,
    #                                   index=X_train_full.index)
    # bert_val_df = pd.DataFrame(bert_val_reduced, columns=bert_train_val_reduced_columns, index=X_val.index)
    # bert_test_df = pd.DataFrame(bert_test_reduced, columns=bert_train_full_reduced_columns, index=X_test.index)

    # X_train_full = pd.concat([X_train_full.drop(columns=bert_columns), bert_train_full_df], axis=1)
    # X_train_val = pd.concat([X_train_val.drop(columns=bert_columns), bert_train_val_df], axis=1)
    # X_val = pd.concat([X_val.drop(columns=bert_columns), bert_val_df], axis=1)
    # X_test = pd.concat([X_test.drop(columns=bert_columns), bert_test_df], axis=1)

    # Scaling Features
    print("Scaling features...")
    columns_not_to_scale = ['Is_Key_Period']
    columns_to_scale = [col for col in X_train_val.columns if col not in columns_not_to_scale]

    scaler = StandardScaler()
    scaler_x_train_full = StandardScaler()
    X_train_full[columns_to_scale] = scaler_x_train_full.fit_transform(X_train_full[columns_to_scale])
    X_test[columns_to_scale] = scaler_x_train_full.transform(X_test[columns_to_scale])
    X_train_val[columns_to_scale] = scaler.fit_transform(X_train_val[columns_to_scale])
    X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])

    return X_train_val, X_val, y_train_val, y_val, X_train_full, y_train_full, X_test

def get_cross_val_scores(model, X, y):
    """
    Get cross-validation scores for a given model.
    """
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Mean Cross validation Accuracy: {scores.mean()}")
    print("Individual Cross validation scores: ", scores)


# Evaluation Function
def evaluate_model(model, X, y):
    """Evaluate a trained model on provided data."""
    predictions = model.predict(X)
    #  if hasattr(model, "predict_proba"):  # Check if model supports probability predictions
    #      probabilities = model.predict_proba(X)[:, 1]
    #  else:
    #     probabilities = None
    threshold = 0.5
    predictions = (predictions > threshold).astype(int)

    # Metrics
    print("Results:")
    print(classification_report(y, predictions))
    #print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    #print(f"F1 Score: {f1_score(y, predictions):.4f}")
    #if probabilities is not None:
    #   print(f"AUC-ROC: {roc_auc_score(y, probabilities):.4f}")

#  return predictions  #, probabilities



aggregated_df_train= pd.read_csv('train_test/aggregated_df_train.csv')
aggregated_df_train.BERT_Embedding  = aggregated_df_train.BERT_Embedding  .apply(string_to_array)
aggregated_df_train.GloVe_Embedding  = aggregated_df_train.GloVe_Embedding  .apply(string_to_array)
aggregated_df_test= pd.read_csv('train_test/aggregated_df_test.csv')
aggregated_df_test.BERT_Embedding = aggregated_df_test.BERT_Embedding.apply(string_to_array)
aggregated_df_test.GloVe_Embedding = aggregated_df_test.GloVe_Embedding.apply(string_to_array)
expanded_df_train, bert_columns, glove_columns = expand_embeddings(aggregated_df_train)
expanded_df_test, _, _ = expand_embeddings(aggregated_df_test)


print (expanded_df_train.shape, expanded_df_test.shape)
aggregated_df_train = aggregated_df_train . drop(columns=['PeriodID', 
                                                        #   ])
                                                          'Sentiment_joy', 'Sentiment_anger','Sentiment_fear','Sentiment_sadness','Sentiment_surprise', 
                                                        #   ],  inplace=False)
                                                          'Tweet_Count', 'Exclamation_Count', 'Question_Count','Uppercase_Ratio', 'Repeated_Char_Word_Ratio'],  inplace=False)



# Preparing data for training
X_train_val, X_val, y_train_val, y_val, X_train_full, y_train_full, X_testing_data = prepare_train_and_test_data(
    expanded_df_train, expanded_df_test, bert_columns, glove_columns
)

input_dim = 1
batch_size = 32
learning_rate = 0.01
num_epochs = 10000
weight_decay = 0.001


model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train_val.shape[1]]),
    layers.Dense(700,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(rate=0.6),
    layers.Dense(200,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(rate=0.6),
    layers.Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),    
    layers.Dense(1,activation='sigmoid'),
])
adam = keras.optimizers.Adam(learning_rate=0.001,weight_decay=0.001)
model.compile(optimizer=adam,    loss='binary_crossentropy',
    metrics=['binary_accuracy'])
early_stopping = EarlyStopping(
    min_delta=0.005,# minimium amount of change to count as an improvement
    patience=25,# how many epochs to wait before stopping
    restore_best_weights=True,
)
model.summary()



model.fit(X_train_val, y_train_val,
    validation_data=(X_val, y_val),
    epochs=num_epochs, batch_size=64,callbacks=[early_stopping])
print(model)
# # Define the threshold

# Convert probabilities to binary class labels
evaluate_model(model, X_train_val, y_train_val)
evaluate_model(model, X_val, y_val)

# nn_predicted_proba = model.predict(X_full)
# y_pred = (nn_predicted_proba > threshold).astype(int)


# # Save the model