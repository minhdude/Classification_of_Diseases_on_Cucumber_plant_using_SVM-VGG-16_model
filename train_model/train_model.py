import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os
from ProcessImage.feature_extraction import extract_features
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC


def plot_metrics_across_folds(f1_scores, accuracies, recalls, precisions):
    plt.figure(figsize=(10, 6))

    plt.plot(f1_scores, label='F1-score', marker='o')
    plt.plot(accuracies, label='Accuracy', marker='o')
    plt.plot(recalls, label='Recall', marker='o')
    plt.plot(precisions, label='Precision', marker='o')

    plt.title('Metrics Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_data(dataset_path):
    labels = os.listdir(dataset_path)
    data = []
    target = []

    for label in labels:
        label_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            features = extract_features(img_path)
            data.append(features)
            target.append(label)

    data = np.array(data)
    target = np.array(target)

    label_to_numeric = {label: idx for idx, label in enumerate(np.unique(target))}
    target = np.array([label_to_numeric[label] for label in target])

    return data, target


def train_test_split_data(data, target, test_size=0.2, random_state=42):
    return train_test_split(data, target, test_size=test_size, random_state=random_state)


def perform_k_fold_cross_validation(data, target, C_value):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_accuracy = 0
    best_model = None
    fold = 1

    f1_scores = []
    accuracies = []
    recalls = []
    precisions = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        svm_model = LinearSVC(multi_class='crammer_singer', C=C_value, max_iter=1000)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        accuracies.append(accuracy)
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)

        print(f"Metrics for Fold {fold}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"Classification report for Fold {fold}:\n{classification_report(y_test, y_pred)}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = svm_model
            print(f"Updating best model for Fold {fold}")

        fold += 1

    return best_model, best_accuracy, f1_scores, accuracies, recalls, precisions


def train_and_evaluate_svm(X_train, X_test, y_train, y_test, C_values):
    best_accuracy = 0
    best_C = None

    for C in C_values:
        svm_model = LinearSVC(multi_class='crammer_singer', C=C, max_iter=1000)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"Accuracy for C={C}: {accuracy}")
        print(f"F1 score for C={C}: {f1}")
        print(f"Precision for C={C}: {precision}")
        print(f"Recall for C={C}: {recall}")
        print(f"Classification report for C={C}:\n{classification_report(y_test, y_pred)}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C

    print(f"Best accuracy: {best_accuracy} achieved with C={best_C}")
    return best_C


def save_best_model(model, accuracy):
    if model is not None:
        best_model_filename = "best_svm_model.joblib"
        joblib.dump(model, best_model_filename)
        print(f"Best model saved as {best_model_filename} with accuracy: {accuracy}")
    else:
        print("No best model found.")


# Now, using these functions:

dataset_path = r"C:\Users\minhd\Desktop\Khai_pha_du_lieu\Khai_pha_du_lieu\Augmented Image"
data, target = load_data(dataset_path)
# Assuming you have performed k-fold cross-validation and obtained these arrays
# f1_scores, accuracies, recalls, precisions = perform_k_fold_cross_validation(data, target, C_value)


X_train, X_test, y_train, y_test = train_test_split_data(data, target)

C_values = [0.001, 0.01, 0.1, 1, 10]
best_C = train_and_evaluate_svm(X_train, X_test, y_train, y_test, C_values)

best_model, best_accuracy, f1_scores, accuracies, recalls, precisions = perform_k_fold_cross_validation(data, target,
                                                                                                        best_C)

plot_metrics_across_folds(f1_scores, accuracies, recalls, precisions)

save_best_model(best_model, best_accuracy)
