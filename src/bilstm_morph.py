import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# PARSER WITH MORPH FEATURES
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
            word  = cols[1]
            pos   = cols[3]
            feats = cols[5]
            if feats == "_":
                feats = "NO_MORPH"
            sentence.append((word, pos, feats))

    if sentence:
        sentences.append(sentence)

    return sentences


# -----------------------------
# BUILD VOCABULARIES
# -----------------------------
def build_vocab(sentences):
    word_to_ix  = {"<PAD>": 0, "<UNK>": 1}
    morph_to_ix = {"<PAD>": 0, "<UNK_MORPH>": 1}
    tag_to_ix   = {"<PAD>": 0}

    for sentence in sentences:
        for word, tag, feats in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if feats not in morph_to_ix:
                morph_to_ix[feats] = len(morph_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, morph_to_ix, tag_to_ix


# -----------------------------
# ENCODE DATA
# -----------------------------
def encode_data(sentences, word_to_ix, morph_to_ix, tag_to_ix):
    encoded = []
    for sentence in sentences:
        words = torch.tensor(
            [word_to_ix.get(w, 1) for w, t, m in sentence], dtype=torch.long
        )
        morphs = torch.tensor(
            [morph_to_ix.get(m, morph_to_ix["<UNK_MORPH>"]) for w, t, m in sentence],
            dtype=torch.long
        )
        tags = torch.tensor(
            [tag_to_ix[t] for w, t, m in sentence], dtype=torch.long
        )
        encoded.append((words, morphs, tags))
    return encoded


def collate_fn(batch):
    words_batch, morphs_batch, tags_batch = zip(*batch)
    words_padded  = pad_sequence(words_batch,  batch_first=True, padding_value=0)
    morphs_padded = pad_sequence(morphs_batch, batch_first=True, padding_value=0)
    tags_padded   = pad_sequence(tags_batch,   batch_first=True, padding_value=0)
    return words_padded, morphs_padded, tags_padded


# -----------------------------
# MODEL
# -----------------------------
class BiLSTMMorph(nn.Module):
    def __init__(self, vocab_size, morph_size, tagset_size,
                 word_emb_dim=128, morph_emb_dim=32, hidden_dim=256):
        super(BiLSTMMorph, self).__init__()

        self.word_emb  = nn.Embedding(vocab_size, word_emb_dim,  padding_idx=0)
        self.morph_emb = nn.Embedding(morph_size, morph_emb_dim, padding_idx=0)
        self.dropout   = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            word_emb_dim + morph_emb_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words, morphs):
        w_emb    = self.word_emb(words)
        m_emb    = self.morph_emb(morphs)
        combined = torch.cat((w_emb, m_emb), dim=2)
        combined = self.dropout(combined)
        lstm_out, _ = self.lstm(combined)
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

    total_batches = len(loader)
    start = time.time()

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, (words, morphs, tags) in enumerate(loader):
            words  = words.to(DEVICE)
            morphs = morphs.to(DEVICE)
            tags   = tags.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(words, morphs)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags    = tags.view(-1)

            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {i+1}/{total_batches} | Loss: {total_loss/(i+1):.4f}")

        print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss/total_batches:.4f}")

    print("Training Time:", round(time.time() - start, 3), "seconds")


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(model, test_data):
    model.eval()
    model.to(DEVICE)

    loader = DataLoader(test_data, batch_size=64,
                        shuffle=False, collate_fn=collate_fn)

    correct = 0
    total   = 0
    pad_ix  = 0

    with torch.no_grad():
        for words, morphs, tags in loader:
            words  = words.to(DEVICE)
            morphs = morphs.to(DEVICE)
            tags   = tags.to(DEVICE)

            outputs     = model(words, morphs)
            predictions = torch.argmax(outputs, dim=-1)

            mask    = tags != pad_ix
            correct += (predictions[mask] == tags[mask]).sum().item()
            total   += mask.sum().item()

    accuracy = correct / total
    print("BiLSTM + Morph Accuracy:", round(accuracy, 4))
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

    word_to_ix, morph_to_ix, tag_to_ix = build_vocab(train_sentences)
    print(f"Vocab: {len(word_to_ix)} words | Morph features: {len(morph_to_ix)} | Tags: {len(tag_to_ix)}")

    train_data = encode_data(train_sentences, word_to_ix, morph_to_ix, tag_to_ix)
    test_data  = encode_data(test_sentences,  word_to_ix, morph_to_ix, tag_to_ix)

    model = BiLSTMMorph(len(word_to_ix), len(morph_to_ix), len(tag_to_ix))
    print("Model ready. Starting training...\n")

    train_model(model, train_data, epochs=3, batch_size=32)

    print("\nEvaluating...")
    evaluate(model, test_data)