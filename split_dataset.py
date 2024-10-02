# Import libraries 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
def read_corpus(corpus_path):
    data = [] # Initialize an empty list to store the data
    with open(corpus_path, encoding = 'utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 3) # Same corpus, still split the line in 4 parts
            
            if len(parts) == 4: 
                category, __, __, text = parts # This week's assignment is multiclass classification, so sentiment and ids are not necessary. 
                
                # data tagging
                data.append(
                    {
                        'category': category,
                        'text': text
                    }
                )
    return pd.DataFrame(data)

# Split the dataset
def split_dataset(data, train_size = 0.7, val_size = 0.15, test_size = 0.15, stratify_col = 'category'):
    # First, split the dataset into training and testing sets
    train_val, test_data = train_test_split(data, test_size = test_size, stratify=data[stratify_col], random_state = 42)
    
    # Split the testing set into validation and testing sets
    relative_test_size = val_size / (train_size + val_size)
    train_data, val_data = train_test_split(train_val, test_size=relative_test_size, stratify=train_val[stratify_col], random_state = 42)
    return train_data, val_data, test_data

# Save the dataset
def save_dataset(train_data, val_data, test_data, output_dir):
    # Save the three datasets in txt files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    for name, dataset in datasets.items():
        with open(os.path.join(output_dir, f'{name}.txt'), 'w', encoding = 'utf-8') as file:
            for _, row in dataset.iterrows():
                file.write(f"{row['category']}\t{row['text']}\n")
                
# Main function
def main(corpus_path, output_dir):
    # Load the dataset
    data = read_corpus(corpus_path)
    
    # Split the dataset
    train_data, val_data, test_data = split_dataset(data)
    
    # Save the dataset
    save_dataset(train_data, val_data, test_data, output_dir)
    
    print(f"Dataset split into train, validation and test sets and saved at {output_dir}")  
    
if __name__ == '__main__':
    # Define the path to the corpus 
    corpus_path = '/Users/hongxuzhou/LfD/LfD_Assignment3/corpus_of_reviews.txt'
    output_dir = '/Users/hongxuzhou/LfD/LfD_Assignment3/Datasets'
    main(corpus_path, output_dir)