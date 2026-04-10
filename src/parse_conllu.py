import sklearn_crfsuite
from sklearn_crfsuite import metrics
import time
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


# -----------------------------
# DATA PARSER
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
            feats = cols[5]

            sentence.append((word, pos, feats))

    if sentence:
        sentences.append(sentence)

    return sentences


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def word2features(sentence, i, mode="full"):
    word, pos, feats = sentence[i]
    features = {}

    # WORD FEATURES
    if mode in ["baseline", "full", "no_gender", "no_number", "no_tense", "no_person"]:
        features["word"] = word
        features["word.lower()"] = word.lower()
        features["is_capitalized"] = word[0].isupper()
        features["is_digit"] = word.isdigit()

    # MORPHOLOGICAL FEATURES
    if mode in ["morph_only", "full", "no_gender", "no_number", "no_tense", "no_person"] and feats != "_":
        for feat in feats.split("|"):
            key_value = feat.split("=")
            if len(key_value) == 2:
                key, value = key_value

                if mode == "no_gender" and key == "Gender":
                    continue
                if mode == "no_number" and key == "Number":
                    continue
                if mode == "no_tense" and key == "Tense":
                    continue
                if mode == "no_person" and key == "Person":
                    continue

                features[f"morph_{key}"] = value

    # CONTEXT
    if i > 0:
        features["prev_word"] = sentence[i - 1][0]
    else:
        features["BOS"] = True

    if i < len(sentence) - 1:
        features["next_word"] = sentence[i + 1][0]
    else:
        features["EOS"] = True

    return features


def sentence2features(sentence, mode="full"):
    return [word2features(sentence, i, mode) for i in range(len(sentence))]


def sentence2labels(sentence):
    return [token[1] for token in sentence]


# -----------------------------
# TRAIN + EVALUATE FUNCTION
# -----------------------------
def train_and_report(train_sentences, test_sentences, mode="full"):

    print(f"\n====== MODE: {mode.upper()} ======")

    X_train = [sentence2features(s, mode) for s in train_sentences]
    y_train = [sentence2labels(s) for s in train_sentences]

    X_test = [sentence2features(s, mode) for s in test_sentences]
    y_test = [sentence2labels(s) for s in test_sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )

    # TRAIN
    start_train = time.time()
    crf.fit(X_train, y_train)

    print("\nTop 20 Most Informative Features:")
    for (attr, label), weight in sorted(
            crf.state_features_.items(),
            key=lambda x: abs(x[1]),
            reverse=True)[:20]:
        print(f"{attr} -> {label}: {weight:.4f}")

    end_train = time.time()

    # TEST
    start_test = time.time()
    y_pred = crf.predict(X_test)
    end_test = time.time()

    accuracy = metrics.flat_accuracy_score(y_test, y_pred)

    print("Training Time:", round(end_train - start_train, 3), "seconds")
    print("Inference Time:", round(end_test - start_test, 3), "seconds")
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:\n")
    print(metrics.flat_classification_report(y_test, y_pred, digits=3))

    # Flatten labels
    y_test_flat = [label for sent in y_test for label in sent]
    y_pred_flat = [label for sent in y_pred for label in sent]

    # Error rate
    total_tokens = len(y_test_flat)
    total_errors = sum(t != p for t, p in zip(y_test_flat, y_pred_flat))

    print("\nTotal Tokens:", total_tokens)
    print("Total Errors:", total_errors)
    print("Error Rate:", round(total_errors / total_tokens, 4))

    return accuracy, y_test_flat, y_pred_flat


# -----------------------------
# MCNEMAR TEST (UPDATED)
# -----------------------------
def run_mcnemar(y_true, y_pred1, y_pred2):

    table = [[0, 0],
             [0, 0]]

    for t, p1, p2 in zip(y_true, y_pred1, y_pred2):
        correct1 = (p1 == t)
        correct2 = (p2 == t)

        if correct1 and correct2:
            table[0][0] += 1
        elif correct1 and not correct2:
            table[0][1] += 1
        elif not correct1 and correct2:
            table[1][0] += 1
        else:
            table[1][1] += 1

    result = mcnemar(table, exact=False, correction=True)

    print("\n====== McNemar Test ======")
    print("Test Statistic:", result.statistic)
    print("p-value:", result.pvalue)

    return result.statistic, result.pvalue


# -----------------------------
# MAIN EXPERIMENT PIPELINE
# -----------------------------
if __name__ == "__main__":

    train_file = "data/en_ewt-ud-train.conllu"
    test_file = "data/en_ewt-ud-test.conllu"

    train_sentences = parse_conllu(train_file)
    test_sentences = parse_conllu(test_file)

    results = []

    # Run models
    baseline_acc, y_true_base, y_pred_base = train_and_report(train_sentences, test_sentences, mode="baseline")
    results.append({"Model": "Baseline", "Accuracy": baseline_acc})

    morph_only_acc, _, _ = train_and_report(train_sentences, test_sentences, mode="morph_only")
    results.append({"Model": "Morph-only", "Accuracy": morph_only_acc})

    full_acc, y_true_full, y_pred_full = train_and_report(train_sentences, test_sentences, mode="full")
    results.append({"Model": "Full", "Accuracy": full_acc})

    no_gender_acc, _, _ = train_and_report(train_sentences, test_sentences, mode="no_gender")
    results.append({"Model": "No_Gender", "Accuracy": no_gender_acc})

    no_number_acc, _, _ = train_and_report(train_sentences, test_sentences, mode="no_number")
    results.append({"Model": "No_Number", "Accuracy": no_number_acc})

    no_tense_acc, _, _ = train_and_report(train_sentences, test_sentences, mode="no_tense")
    results.append({"Model": "No_Tense", "Accuracy": no_tense_acc})

    no_person_acc, _, _ = train_and_report(train_sentences, test_sentences, mode="no_person")
    results.append({"Model": "No_Person", "Accuracy": no_person_acc})

    # McNemar Test
    stat, pval = run_mcnemar(y_true_base, y_pred_base, y_pred_full)
    results.append({
        "Model": "McNemar_Full_vs_Baseline",
        "Accuracy": f"Statistic={stat}, p-value={pval}"
    })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)

    print("\nResults saved to experiment_results.csv")