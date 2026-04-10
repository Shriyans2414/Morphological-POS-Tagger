import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# DATA LOADER
# -----------------------------
def parse_conllu(file_path):
    sentences = []
    sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            word = cols[1]
            pos = cols[3]
            sentence.append((word, pos))

    if sentence:
        sentences.append(sentence)

    return sentences


def build_vocab(sentences):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    tag_to_ix = {"<PAD>": 0}

    for sentence in sentences:
        for word, tag in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


def encode_data(sentences, word_to_ix, tag_to_ix):
    encoded = []
    for sentence in sentences:
        words = torch.tensor(
            [word_to_ix.get(w, 1) for w, t in sentence], dtype=torch.long
        )
        tags = torch.tensor(
            [tag_to_ix[t] for w, t in sentence], dtype=torch.long
        )
        encoded.append((words, tags))
    return encoded


def collate_fn(batch):
    words_batch, tags_batch = zip(*batch)
    words_padded = pad_sequence(words_batch, batch_first=True, padding_value=0)
    tags_padded  = pad_sequence(tags_batch,  batch_first=True, padding_value=0)
    return words_padded, tags_padded


# -----------------------------
# MODEL
# -----------------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size,
                 embedding_dim=128, hidden_dim=256):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout   = nn.Dropout(0.3)
        self.lstm      = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


# -----------------------------
# TRAIN
# -----------------------------
def train_model(model, train_data, epochs=3, batch_size=32):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    loader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_fn)

    start = time.time()
    total_batches = len(loader)

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, (words, tags) in enumerate(loader):
            words = words.to(DEVICE)
            tags  = tags.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(words)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags    = tags.view(-1)

            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print progress every 50 batches
            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {i+1}/{total_batches} | Loss: {total_loss/(i+1):.4f}")

        print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss/total_batches:.4f}")

    print("Training Time:", round(time.time() - start, 3), "seconds")


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(model, test_data, tag_to_ix):
    model.eval()
    model.to(DEVICE)

    loader = DataLoader(test_data, batch_size=64,
                        shuffle=False, collate_fn=collate_fn)

    correct = 0
    total   = 0
    pad_ix  = 0

    with torch.no_grad():
        for words, tags in loader:
            words = words.to(DEVICE)
            tags  = tags.to(DEVICE)

            outputs     = model(words)
            predictions = torch.argmax(outputs, dim=-1)

            mask = tags != pad_ix
            correct += (predictions[mask] == tags[mask]).sum().item()
            total   += mask.sum().item()

    accuracy = correct / total
    print("BiLSTM Accuracy:", round(accuracy, 4))
    return accuracy


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    train_file = "data/en_ewt-ud-train.conllu"
    test_file  = "data/en_ewt-ud-test.conllu"

    print("Loading data...")
    train_sentences = parse_conllu(train_file)
    test_sentences  = parse_conllu(test_file)
    print(f"Train: {len(train_sentences)} sentences | Test: {len(test_sentences)} sentences")

    word_to_ix, tag_to_ix = build_vocab(train_sentences)
    print(f"Vocab size: {len(word_to_ix)} | Tags: {len(tag_to_ix)}")

    train_data = encode_data(train_sentences, word_to_ix, tag_to_ix)
    test_data  = encode_data(test_sentences,  word_to_ix, tag_to_ix)

    model = BiLSTMTagger(len(word_to_ix), len(tag_to_ix))
    print("Model ready. Starting training...\n")

    train_model(model, train_data, epochs=3, batch_size=32)

    print("\nEvaluating...")
    evaluate(model, test_data, tag_to_ix)