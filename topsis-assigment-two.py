import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Function to load a dataset from the datasets library
def load_dataset_from_library(dataset_name):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    return df

# Function to train and evaluate a model on a dataset
def train_and_evaluate_model(model_name, train_data, test_data):
    print(f"Processing {model_name}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize and preprocess the training and testing data
    train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True, return_tensors='pt')

    # Create Dataset objects
    train_dataset = list(zip(train_encodings['input_ids'],
                             train_encodings['attention_mask'],
                             train_data['label']))

    test_dataset = list(zip(test_encodings['input_ids'],
                            test_encodings['attention_mask'],
                            test_data['label']))

    # Define Trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        output_dir="./models",  # Change this path as needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.predictions.argmax(axis=1), p.label_ids),
            'precision': precision_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
            'recall': recall_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
            'f1': f1_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
        }
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    return results

# Download and use the IMDb dataset as an example
dataset_name = "imdb"
df = load_dataset_from_library(dataset_name)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# List of models to try
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "xlnet-base-cased",
    "albert-base-v2"
]

# List to store evaluation metrics for each model
metrics_list = []

# Loop through each model
for model_name in model_names:
    results = train_and_evaluate_model(model_name, train_data, test_data)
    
    # Append metrics to the list
    metrics_list.append({
        'Model': model_name,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1_Score': results['f1'],
    })

# Create a DataFrame from the list
metrics_df = pd.DataFrame(metrics_list)

# Save the DataFrame to a CSV file
output_file_path = 'output.csv'
metrics_df.to_csv(output_file_path, index=False)

print(f"Metrics saved to {output_file_path}")
