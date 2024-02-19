import json
import numpy as np
from src.nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.model import NeuralNetwork
import logging


logging.basicConfig(filename='chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open("intents.json", "r") as fin:
        intents = json.load(fin)
except OSError as err_1:
    logging.error(f"Error loading intents.json: {err_1}")
    exit(1)
    
all_words = []
tags = []
xy = []

# Extracting data from intents
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Generating training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    """
    Dataset for the ChatBot.
    """
    def __init__(self):
        super().__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 2000

    # Creating dataset and data loader
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    # Setting device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    try:
        for epoch in range(num_epochs):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device).long()

                output = model(words)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss={loss.item():.4f}")

        print(f"Final loss={loss.item():.4f}")
    except RuntimeError as err_2:
        logging.critical(f"Error during training: {err_2}")

    # Saving trained model and related data
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    
    try:
        torch.save(data, FILE)
        print(f"Training complete! File saved to {FILE}")
    except OSError as err_3:
        logging.critical(f"Error saving model: {err_3}")
        print(f"Error saving model: {err_3}")
