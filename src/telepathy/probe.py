from __future__ import annotations

import argparse
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from telepathy.common import ensureParent, loadConfig, readJsonl, writeJson


def scorePredictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if len(np.unique(labels)) == 2 else None,
    }


def pairClusterBootstrap(labels: np.ndarray, predictions: np.ndarray, pair_ids: list[str], seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed)
    unique_pairs = np.unique(pair_ids)
    pair_indices = [np.flatnonzero(np.asarray(pair_ids) == pair_id) for pair_id in unique_pairs]
    accuracies = []
    balanced = []
    for _ in range(count):
        sampled_pairs = rng.integers(0, len(pair_indices), size=len(pair_indices))
        selected = np.concatenate([pair_indices[index] for index in sampled_pairs])
        bootstrap_labels = labels[selected]
        bootstrap_predictions = predictions[selected]
        accuracies.append(accuracy_score(bootstrap_labels, bootstrap_predictions))
        if len(np.unique(bootstrap_labels)) == 2:
            balanced.append(balanced_accuracy_score(bootstrap_labels, bootstrap_predictions))
    return {
        "accuracy_95_ci": [float(value) for value in np.quantile(accuracies, [0.025, 0.975])],
        "balanced_accuracy_95_ci": [float(value) for value in np.quantile(balanced, [0.025, 0.975])] if balanced else None,
        "bootstrap_unit": "paired example pair_id",
    }


def writeViewer(path: str, rows: list[dict], predictions: list[str], layer: int) -> None:
    cards = []
    for row, prediction in zip(rows, predictions):
        cards.append(
            "<article><h2>" + html.escape(row["example_id"]) + "</h2>"
            "<p><strong>Public task:</strong> " + html.escape(row["public_task"]) + "</p>"
            "<p><strong>Public response (forced text):</strong> " + html.escape(row["forced_public_response"]) + "</p>"
            "<p><strong>Monitor prediction:</strong> " + html.escape(prediction) + "</p>"
            "<p><strong>Revealed action:</strong> " + html.escape(row["actual_choice"]) + "</p></article>"
        )
    document = """<!doctype html><meta charset="utf-8"><title>LLM Telepathy examples</title>
<style>body{font:16px system-ui;max-width:850px;margin:2rem auto;padding:0 1rem;background:#101821;color:#edf4f4}
article{background:#1b2b34;border:1px solid #38515b;border-radius:12px;padding:1rem 1.3rem;margin:1rem 0}
h1{color:#a8e6cf}strong{color:#f8c471}</style>
<h1>LLM Telepathy: held-out examples</h1><p>Selected layer: LAYER. The public response is fixed text.</p>CARDS"""
    document = document.replace("LAYER", str(layer)).replace("CARDS", "\n".join(cards))
    ensureParent(path)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit layerwise linear probes and report held-out metrics.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = loadConfig(args.config)
    with open(config["paths"]["run_metadata"], encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    outcomes_path = metadata["outcomes_file"]
    rows = readJsonl(outcomes_path)
    activations = np.load(config["paths"]["activations"])
    valid_indices = [index for index, row in enumerate(rows) if row["actual_choice"] in ("A", "B")]
    valid_rows = [rows[index] for index in valid_indices]
    if not any(row["split"] == "train" and row["actual_choice"] in ("A", "B") for row in rows):
        raise RuntimeError("No valid A/B outcomes in the training split; see outcomes.jsonl for refusal and malformed generations.")
    if not any(row["split"] == "validation" and row["actual_choice"] in ("A", "B") for row in rows):
        raise RuntimeError("No valid A/B outcomes in the validation split; see outcomes.jsonl for refusal and malformed generations.")
    if not any(row["split"] == "test" and row["actual_choice"] in ("A", "B") for row in rows):
        raise RuntimeError("No valid A/B outcomes in the test split; see outcomes.jsonl for refusal and malformed generations.")
    labels = np.array([1 if row["actual_choice"] == "B" else 0 for row in valid_rows], dtype=np.int64)
    train_mask = np.array([row["split"] == "train" for row in valid_rows])
    validation_mask = np.array([row["split"] == "validation" for row in valid_rows])
    test_mask = np.array([row["split"] == "test" for row in valid_rows])
    test_pair_ids = [row["pair_id"] for row in valid_rows if row["split"] == "test"]
    c_values = config["probe"]["c_values"]
    layer_count = metadata["layer_count"]
    layer_results = []
    layer_predictions = []
    selected = None

    for layer_index in range(layer_count):
        features = activations[f"layer_{layer_index}"][valid_indices]
        scaler = StandardScaler().fit(features[train_mask])
        scaled = scaler.transform(features)
        candidates = []
        if len(np.unique(labels[train_mask])) < 2:
            classifier = DummyClassifier(strategy="constant", constant=int(labels[train_mask][0]))
            classifier.fit(scaled[train_mask], labels[train_mask])
            validation_predictions = classifier.predict(scaled[validation_mask])
            validation_score = (balanced_accuracy_score(labels[validation_mask], validation_predictions)
                                if len(np.unique(labels[validation_mask])) == 2
                                else accuracy_score(labels[validation_mask], validation_predictions))
            best_validation_score, best_c, best_classifier = float(validation_score), None, classifier
            validation_metric = "balanced_accuracy" if len(np.unique(labels[validation_mask])) == 2 else "accuracy_single_validation_class"
        else:
            for c_value in c_values:
                classifier = LogisticRegression(C=float(c_value), max_iter=2000, random_state=int(config["probe"]["seed"]))
                classifier.fit(scaled[train_mask], labels[train_mask])
                validation_predictions = classifier.predict(scaled[validation_mask])
                validation_score = (balanced_accuracy_score(labels[validation_mask], validation_predictions)
                                    if len(np.unique(labels[validation_mask])) == 2
                                    else accuracy_score(labels[validation_mask], validation_predictions))
                candidates.append((float(validation_score), float(c_value), classifier))
            # Deterministic tie break favors stronger regularization (smaller C).
            best_validation_score, best_c, best_classifier = max(candidates, key=lambda item: (item[0], -item[1]))
            validation_metric = "balanced_accuracy" if len(np.unique(labels[validation_mask])) == 2 else "accuracy_single_validation_class"
        test_predictions = best_classifier.predict(scaled[test_mask])
        test_labels = labels[test_mask]
        scores = scorePredictions(test_labels, test_predictions)
        scores.update(pairClusterBootstrap(test_labels, test_predictions, test_pair_ids,
                                           int(config["probe"]["seed"]) + layer_index,
                                           int(config["probe"]["bootstrap_samples"])))
        layer_results.append({
            "layer": layer_index,
            "selected_c": best_c,
            "fit_status": "linear_probe" if best_c is not None else "constant_fallback_train_has_one_class",
            "validation_selection_score": best_validation_score,
            "validation_selection_metric": validation_metric,
            "validation_balanced_accuracy": (best_validation_score if validation_metric == "balanced_accuracy" else None),
            "test": scores,
        })
        layer_predictions.append((test_predictions, best_classifier, scaler))
        if selected is None or best_validation_score > selected["validation_selection_score"]:
            selected = {
                "layer": layer_index,
                "validation_selection_score": best_validation_score,
                "validation_selection_metric": validation_metric,
                "selected_c": best_c,
            }

    chosen_predictions = layer_predictions[selected["layer"]][0]
    test_rows = [row for row in valid_rows if row["split"] == "test"]
    test_labels = labels[test_mask]
    train_labels = labels[train_mask]
    majority = int(np.bincount(train_labels, minlength=2).argmax())
    majority_predictions = np.full_like(test_labels, majority)
    shuffled_rng = np.random.default_rng(int(config["probe"]["seed"]))
    shuffled_train_labels = train_labels.copy()
    shuffled_rng.shuffle(shuffled_train_labels)
    shuffled_feature_index = selected["layer"]
    shuffled_features = activations[f"layer_{shuffled_feature_index}"][valid_indices]
    shuffled_scaler = StandardScaler().fit(shuffled_features[train_mask])
    shuffled_scaled = shuffled_scaler.transform(shuffled_features)
    if len(np.unique(shuffled_train_labels)) < 2 or selected["selected_c"] is None:
        shuffled_classifier = DummyClassifier(strategy="most_frequent")
    else:
        shuffled_classifier = LogisticRegression(C=selected["selected_c"], max_iter=2000,
                                                  random_state=int(config["probe"]["seed"]))
    shuffled_classifier.fit(shuffled_scaled[train_mask], shuffled_train_labels)
    shuffled_predictions = shuffled_classifier.predict(shuffled_scaled[test_mask])

    class_counts = {
        split: {
            "A": sum(row["actual_choice"] == "A" for row in rows if row["split"] == split),
            "B": sum(row["actual_choice"] == "B" for row in rows if row["split"] == split),
            "malformed": sum(row["actual_choice"] == "malformed" for row in rows if row["split"] == split),
            "refusal": sum(row["actual_choice"] == "refusal" for row in rows if row["split"] == split),
        }
        for split in ("train", "validation", "test")
    }
    instruction_rates = {
        split: {
            "valid_action_count": sum(row["split"] == split and row["actual_choice"] in ("A", "B") for row in rows),
            "followed_intended": sum(row["split"] == split and row["instruction_followed"] for row in rows),
            "valid_choice_instruction_failures": sum(row["split"] == split and row["actual_choice"] in ("A", "B") and not row["instruction_followed"] for row in rows),
            "refusals": sum(row["split"] == split and row["actual_choice"] == "refusal" for row in rows),
            "malformed_outputs": sum(row["split"] == split and row["actual_choice"] == "malformed" for row in rows),
            "rate": (sum(row["split"] == split and row["instruction_followed"] for row in rows) /
                     max(1, sum(row["split"] == split and row["actual_choice"] in ("A", "B") for row in rows))),
        }
        for split in ("train", "validation", "test")
    }
    exact_pair_balance = {}
    by_pair = {}
    for row in rows:
        if row["split"] == "test":
            by_pair.setdefault(row["pair_id"], []).append(row["actual_choice"])
    exact_pair_balance = {"pairs": len(by_pair), "both_actual_outcomes": sum(set(outcomes) == {"A", "B"} for outcomes in by_pair.values())}
    metrics = {
        "label_definition": "model-generated actual choice after the forced public response; malformed outputs are excluded from probe fitting and counted separately",
        "probe_config": config["probe"],
        "template_family_splits": config["dataset"]["splits"],
        "valid_binary_examples": len(valid_rows),
        "all_examples_including_refusals_and_malformed": len(rows),
        "class_counts": class_counts,
        "instruction_following": instruction_rates,
        "test_pair_balance": exact_pair_balance,
        "external_monitor": {
            "theoretical_accuracy_if_actual_outcomes_are_balanced_for_identical_public_inputs": 0.5,
            "test_majority_baseline": scorePredictions(test_labels, majority_predictions),
            "test_empirical_class_balance": {"A": int(np.sum(test_labels == 0)), "B": int(np.sum(test_labels == 1))},
            "note": "50% follows from identical public inputs with balanced actual outcomes; it is not an empirical discovery.",
        },
        "controls": {
            "majority_class_test": scorePredictions(test_labels, majority_predictions),
            "shuffled_training_labels_test": scorePredictions(test_labels, shuffled_predictions),
        },
        "selected_layer_by_validation": selected,
        "layer_results": layer_results,
    }
    writeJson(config["paths"]["metrics"], metrics)

    ensureParent(config["paths"]["plot"])
    layer_indices = [result["layer"] for result in layer_results]
    accuracy = [result["test"]["accuracy"] for result in layer_results]
    balanced_accuracy = [result["test"]["balanced_accuracy"] if result["test"]["balanced_accuracy"] is not None else np.nan for result in layer_results]
    lower = [result["test"]["balanced_accuracy_95_ci"][0] if result["test"]["balanced_accuracy_95_ci"] else np.nan for result in layer_results]
    upper = [result["test"]["balanced_accuracy_95_ci"][1] if result["test"]["balanced_accuracy_95_ci"] else np.nan for result in layer_results]
    plt.figure(figsize=(10, 5.5))
    plt.plot(layer_indices, accuracy, marker="o", label="Accuracy")
    plt.plot(layer_indices, balanced_accuracy, marker="s", label="Balanced accuracy")
    plt.fill_between(layer_indices, lower, upper, alpha=0.15, label="Balanced accuracy 95% bootstrap CI")
    plt.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="50% reference")
    plt.xlabel("Transformer layer (decoder block output)")
    plt.ylabel("Held-out test score")
    plt.ylim(0, 1)
    plt.title("Linear probe of the forced public-response position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config["paths"]["plot"], dpi=160)
    plt.close()
    predicted_choices = ["B" if value == 1 else "A" for value in chosen_predictions]
    writeViewer(config["paths"]["viewer"], test_rows, predicted_choices, selected["layer"])
    print(json.dumps({"metrics": config["paths"]["metrics"], "plot": config["paths"]["plot"],
                      "viewer": config["paths"]["viewer"], "selected_layer": selected["layer"]}, indent=2))


if __name__ == "__main__":
    main()
