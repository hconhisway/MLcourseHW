import csv
import math
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random

def weighted_entropy(labels, weights):
    label_weight_sum = defaultdict(float)
    total_weight = sum(weights)

    for label, weight in zip(labels, weights):
        label_weight_sum[label] += weight

    weighted_entropy = 0
    for label, weight_sum in label_weight_sum.items():
        p_label = weight_sum / total_weight
        weighted_entropy -= p_label * np.log2(p_label + 1e-10)

    return weighted_entropy


def weighted_majority_error(labels, weights):
    label_weight_sum = defaultdict(float)
    total_weight = sum(weights)

    for label, weight in zip(labels, weights):
        label_weight_sum[label] += weight

    max_weight = max(label_weight_sum.values())

    return 1 - (max_weight / total_weight)


def weighted_gini_index(labels, weights):
    label_weight_sum = defaultdict(float)
    total_weight = sum(weights)

    for label, weight in zip(labels, weights):
        label_weight_sum[label] += weight

    weighted_gini = 1
    for label, weight_sum in label_weight_sum.items():
        p_label = weight_sum / total_weight
        weighted_gini -= p_label ** 2

    return weighted_gini


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def majority_error(labels):
    total = len(labels)
    counts = Counter(labels)
    majority = counts.most_common(1)[0][1]
    return 1 - (majority / total)


def gini_index(labels):
    total = len(labels)
    counts = Counter(labels)
    return 1 - sum((count / total) ** 2 for count in counts.values())


def compute_medians(dataset, numerical_attrs):
    medians = {}
    for attr in numerical_attrs:
        values = [float(row[attr]) for row in dataset if row[attr] != 'unknown']
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 1:
            median = sorted_values[n // 2]
        else:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        medians[attr] = median
    return medians


def compute_majority_values(dataset, categorical_attrs):
    majority_values = {}
    for attr in categorical_attrs:
        values = [row[attr] for row in dataset if row[attr] != 'unknown']
        majority_values[attr] = Counter(values).most_common(1)[0][0]
    return majority_values


class DecisionTreeNode:
    def __init__(self, attribute=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.children = {}
        self.is_leaf = is_leaf
        self.prediction = prediction


class DecisionTreeNode:
    def __init__(self, attribute=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.children = {}
        self.is_leaf = is_leaf
        self.prediction = prediction


class DecisionTree:
    def __init__(self, criterion='entropy'):
        self.criterion = criterion
        self.tree = None

    def fit(self, data, attributes, weights):
        self.tree = self._build_tree(data, attributes, weights)

    def _best_attribute(self, data, attributes, weights):
        base_criterion_value = self._weighted_criterion_value([row[-1] for row in data], weights)
        best_gain = -1
        best_attr = None
        for attr in attributes:
            subsets = defaultdict(list)
            subset_weights = defaultdict(list)
            for i, row in enumerate(data):
                subsets[row[attr]].append(row)
                subset_weights[row[attr]].append(weights[i])
            weighted_criterion = 0
            for subset_key in subsets:
                subset = subsets[subset_key]
                w_subset = subset_weights[subset_key]
                weighted_criterion += (sum(w_subset) / sum(weights)) * self._weighted_criterion_value(
                    [row[-1] for row in subset], w_subset)
            gain = base_criterion_value - weighted_criterion
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr

    def _weighted_criterion_value(self, labels, weights):
        if self.criterion == 'entropy':
            return weighted_entropy(labels, weights)
        elif self.criterion == 'majority_error':
            return weighted_majority_error(labels, weights)
        elif self.criterion == 'gini_index':
            return weighted_gini_index(labels, weights)

    def _build_tree(self, data, attributes, weights):
        labels = [row[-1] for row in data]
        if len(set(labels)) == 1:
            return DecisionTreeNode(is_leaf=True, prediction=labels[0])

        if not attributes:
            majority_label = self._weighted_majority_label(labels, weights)
            return DecisionTreeNode(is_leaf=True, prediction=majority_label)

        best_attr = self._best_attribute(data, attributes, weights)
        if best_attr is None:
            majority_label = self._weighted_majority_label(labels, weights)
            return DecisionTreeNode(is_leaf=True, prediction=majority_label)

        node = DecisionTreeNode(attribute=best_attr)
        attr_values = set(row[best_attr] for row in data)

        new_attributes = attributes[:]
        new_attributes.remove(best_attr)

        for value in attr_values:
            subset = [row for row in data if row[best_attr] == value]
            subset_weights = [weights[i] for i, row in enumerate(data) if row[best_attr] == value]
            if not subset:
                majority_label = self._weighted_majority_label(labels, weights)
                node.children[value] = DecisionTreeNode(is_leaf=True, prediction=majority_label)
            else:
                child_node = self._build_tree(subset, new_attributes, subset_weights)
                node.children[value] = child_node

        return node

    def _weighted_majority_label(self, labels, weights):
        weighted_counts = defaultdict(float)
        for label, weight in zip(labels, weights):
            weighted_counts[label] += weight
        return max(weighted_counts, key=weighted_counts.get)

    def predict(self, row):
        node = self.tree
        while not node.is_leaf:
            value = row[node.attribute]
            if value in node.children:
                node = node.children[value]
            else:
                return None
        return node.prediction

    def evaluate(self, data):
        predictions = []
        for row in data:
            pred = self.predict(row)
            if pred is None:
                labels = [r[-1] for r in data]
                pred = Counter(labels).most_common(1)[0][0]
            predictions.append(pred)
        actual = [row[-1] for row in data]
        errors = sum(p != a for p, a in zip(predictions, actual))
        return errors / len(data)


from collections import defaultdict, Counter


class AdaBoost:
    def __init__(self, base_learner, T=50):
        self.base_learner = base_learner
        self.T = T
        self.alphas = [] 
        self.learners = []  

    def fit(self, data, attributes, labels):
        n_samples = len(data)
        numeric_labels = np.array([1 if label == 'yes' else -1 for label in labels])

        weights = np.ones(n_samples) / n_samples

        for t in range(self.T):
            learner = self.base_learner(criterion='entropy')
            learner.fit(data, attributes, weights)
            predictions = np.array([1 if learner.predict(row) == 'yes' else -1 for row in data])

            incorrect = (predictions != numeric_labels)
            weighted_error = np.dot(weights, incorrect) / np.sum(weights)

            if weighted_error == 0:
                alpha = 1e10
            elif weighted_error == 1:
                alpha = -1e10
            else:
                alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))

            self.learners.append(learner)
            self.alphas.append(alpha)

            weights *= np.exp(-alpha * numeric_labels * predictions)
            weights /= np.sum(weights)

    def predict(self, data):
        n_samples = len(data)
        final_predictions = np.zeros(n_samples)

        for alpha, learner in zip(self.alphas, self.learners):
            predictions = np.array([1 if learner.predict(row) == 'yes' else -1 for row in data])
            final_predictions += alpha * predictions

        return np.sign(final_predictions)

    def evaluate(self, data, labels):
        numeric_labels = np.array([1 if label == 'yes' else -1 for label in labels])

        predictions = self.predict(data)

        return np.mean(predictions != numeric_labels)


# def load_data(filename):
#     dataset = []
#     with open(filename, 'r') as f:
#         for line in f:
#             terms = line.strip().split(',')
#             dataset.append(terms)
#     return dataset


def load_data(filename, numerical_attrs, categorical_attrs, medians=None, majority_values=None, handle_unknown='value'):
    dataset = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for idx in range(len(row) - 1):
                if row[idx] == 'unknown':
                    if handle_unknown == 'value':
                        pass
                    elif handle_unknown == 'missing':
                        if idx in numerical_attrs:
                            row[idx] = medians[idx]
                        else:
                            row[idx] = majority_values[idx]
                else:
                    if idx in numerical_attrs:
                        row[idx] = float(row[idx])
            for idx in numerical_attrs:
                if row[idx] == 'unknown':
                    row[idx] = medians[idx]
                value = row[idx]
                median = medians[idx]
                row[idx] = 'greater' if value > median else 'less_or_equal'
            dataset.append(row)
    return dataset


def detect_attribute_types(dataset):
    numerical_attrs = set()
    categorical_attrs = set()
    num_attributes = len(dataset[0])
    for attr_idx in range(num_attributes):
        is_numerical = True
        for row in dataset:
            value = row[attr_idx]
            if value == 'unknown':
                continue
            try:
                float(value)
            except ValueError:
                is_numerical = False
                break
        if is_numerical:
            numerical_attrs.add(attr_idx)
        else:
            categorical_attrs.add(attr_idx)
    return numerical_attrs, categorical_attrs


class BaggedTrees:
    def __init__(self, num_trees=10, criterion='entropy'):
        self.num_trees = num_trees
        self.criterion = criterion
        self.trees = []
        self.majority_label = None

    def fit(self, data, attributes, n_jobs=-1):
        n_samples = len(data)
        self.majority_label = Counter([row[-1] for row in data]).most_common(1)[0][0]

        def train_single_tree(seed):
            # Create a new Random instance for this thread
            rng = random.Random(seed)
            # Generate bootstrap sample
            bootstrap_sample = [rng.choice(data) for _ in range(n_samples)]
            # Set uniform weights for bootstrap sample
            weights = [1.0 / n_samples] * n_samples
            # Initialize and train the DecisionTree
            tree = DecisionTree(criterion=self.criterion)
            tree.fit(bootstrap_sample, attributes, weights)
            return tree

        # Generate a list of seeds to ensure different randomness in each parallel process
        seeds = [random.randint(0, 1e6) for _ in range(self.num_trees)]

        # Train trees in parallel
        self.trees = Parallel(n_jobs=n_jobs)(
            delayed(train_single_tree)(seed) for seed in seeds
        )

    def predict(self, row):
        predictions = []
        for tree in self.trees:
            pred = tree.predict(row)
            if pred is not None:
                predictions.append(pred)
        if predictions:
            # Majority vote
            majority_label = Counter(predictions).most_common(1)[0][0]
            return majority_label
        else:
            # If no predictions, default to majority label
            return self.majority_label

    def evaluate(self, data):
        predictions = []
        for row in data:
            pred = self.predict(row)
            predictions.append(pred)
        actual = [row[-1] for row in data]
        errors = sum(p != a for p, a in zip(predictions, actual))
        return errors / len(data)


if __name__ == "__main__":
    AdaBoost = False
    Bagged = False
    Compare = True
    if AdaBoost:
        total_attrs = 16
        attribute_indices = list(range(total_attrs))

        training_data_raw = []
        with open('bank/train.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                training_data_raw.append(row)

        numerical_attrs, categorical_attrs = detect_attribute_types(training_data_raw)
        attribute_indices = list(range(len(training_data_raw[0]) - 1))  # Exclude label

        medians = compute_medians(training_data_raw, numerical_attrs)
        majority_values = compute_majority_values(training_data_raw, categorical_attrs)

        print("\nExperiment (a): Treat 'unknown' as a separate value")

        train_data_a = load_data('bank/train.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                 handle_unknown='value')
        test_data_a = load_data('bank/test.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                handle_unknown='value')

        train_labels = np.array([row[-1] for row in train_data_a])
        test_labels = np.array([row[-1] for row in test_data_a])

        iterations = range(1, 501)
        train_errors = []
        test_errors = []
        stump_train_errors = []
        stump_test_errors = []
        t = 0
        for T in iterations:
            print(t)
            t += 1
            ada = AdaBoost(base_learner=DecisionTree, T=T)
            ada.fit(train_data_a, attribute_indices, train_labels)

            train_error = ada.evaluate(train_data_a, train_labels)
            test_error = ada.evaluate(test_data_a, test_labels)
            train_errors.append(train_error)
            test_errors.append(test_error)

            stump_train_error = []
            stump_test_error = []
            for t, learner in enumerate(ada.learners):
                stump_train_predictions = np.array([learner.predict(row) for row in train_data_a])
                stump_test_predictions = np.array([learner.predict(row) for row in test_data_a])
                stump_train_error.append(np.mean(stump_train_predictions != train_labels))
                stump_test_error.append(np.mean(stump_test_predictions != test_labels))

            stump_train_errors.append(stump_train_error)
            stump_test_errors.append(stump_test_error)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_errors, label='Training Error')
        plt.plot(iterations, test_errors, label='Test Error')
        plt.xlabel('Number of Boosting Iterations (T)')
        plt.ylabel('Error Rate')
        plt.title('Training and Test Errors as a Function of T')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        for i in range(500):
            plt.plot(range(1, len(stump_train_errors[i]) + 1), stump_train_errors[i], label=f'Stump {i + 1} Train',
                     alpha=0.2)
            plt.plot(range(1, len(stump_test_errors[i]) + 1), stump_test_errors[i], label=f'Stump {i + 1} Test', alpha=0.2)

        plt.xlabel('Iteration (Stump)')
        plt.ylabel('Error Rate')
        plt.title('Training and Test Errors of Each Decision Stump')
        plt.grid(True)
        plt.show()
    elif Bagged:
        total_attrs = 16
        attribute_indices = list(range(total_attrs))

        training_data_raw = []
        with open('bank/train.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                training_data_raw.append(row)

        numerical_attrs, categorical_attrs = detect_attribute_types(training_data_raw)
        attribute_indices = list(range(len(training_data_raw[0]) - 1))  # Exclude label

        medians = compute_medians(training_data_raw, numerical_attrs)
        majority_values = compute_majority_values(training_data_raw, categorical_attrs)

        print("\nExperiment (a): Treat 'unknown' as a separate value")

        train_data_a = load_data('bank/train.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                 handle_unknown='value')
        test_data_a = load_data('bank/test.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                handle_unknown='value')

        train_labels = np.array([row[-1] for row in train_data_a])
        test_labels = np.array([row[-1] for row in test_data_a])

        num_trees_range = range(1, 501, 10)
        train_errors = []
        test_errors = []

        for num_trees in num_trees_range:
            print(f"Training Bagged Trees with {num_trees} trees...")

            bagged_trees = BaggedTrees(num_trees=num_trees, criterion='entropy')

            bagged_trees.fit(train_data_a, attribute_indices, n_jobs=-1)

            train_error = bagged_trees.evaluate(train_data_a)
            test_error = bagged_trees.evaluate(test_data_a)

            train_errors.append(train_error)
            test_errors.append(test_error)

        plt.figure(figsize=(10, 6))
        plt.plot(num_trees_range, train_errors, label='Training Error', marker='o')
        plt.plot(num_trees_range, test_errors, label='Test Error', marker='o')
        plt.xlabel('Number of Trees in Bagging Ensemble')
        plt.ylabel('Error Rate')
        plt.title('Training and Test Errors as a Function of Number of Trees')
        plt.legend()
        plt.show()
    elif Compare:
        total_attrs = 16
        attribute_indices = list(range(total_attrs))

        training_data_raw = []
        with open('bank/train.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                training_data_raw.append(row)

        numerical_attrs, categorical_attrs = detect_attribute_types(training_data_raw)
        attribute_indices = list(range(len(training_data_raw[0]) - 1))

        medians = compute_medians(training_data_raw, numerical_attrs)
        majority_values = compute_majority_values(training_data_raw, categorical_attrs)

        train_data_a = load_data('bank/train.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                 handle_unknown='value')
        test_data_a = load_data('bank/test.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                                handle_unknown='value')

        train_labels = np.array([row[-1] for row in train_data_a])
        test_labels = np.array([row[-1] for row in test_data_a])
        test_labels_numeric = np.array([1 if label == 'yes' else -1 for label in test_labels])

        single_tree_predictions = [[] for _ in range(len(test_data_a))]
        bagged_predictions = [[] for _ in range(len(test_data_a))]

        num_runs = 100
        num_trees = 500
        sample_size = 1000

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")

            sample_indices = random.sample(range(len(train_data_a)), sample_size)
            sample_data = [train_data_a[i] for i in sample_indices]

            bagged_trees = BaggedTrees(num_trees=num_trees, criterion='entropy')
            bagged_trees.fit(sample_data, attribute_indices, n_jobs=-1)

            single_tree = bagged_trees.trees[0]

            for idx, test_example in enumerate(test_data_a):
                pred = single_tree.predict(test_example)
                pred_numeric = 1 if pred == 'yes' else -1
                single_tree_predictions[idx].append(pred_numeric)

            for idx, test_example in enumerate(test_data_a):
                pred = bagged_trees.predict(test_example)
                pred_numeric = 1 if pred == 'yes' else -1
                bagged_predictions[idx].append(pred_numeric)

        biases_single = []
        variances_single = []
        for idx in range(len(test_data_a)):
            preds = np.array(single_tree_predictions[idx])
            mean_pred = np.mean(preds)
            bias = (mean_pred - test_labels_numeric[idx]) ** 2
            variance = np.var(preds)
            biases_single.append(bias)
            variances_single.append(variance)

        avg_bias_single = np.mean(biases_single)
        avg_variance_single = np.mean(variances_single)
        mse_single = avg_bias_single + avg_variance_single

        biases_bagged = []
        variances_bagged = []
        for idx in range(len(test_data_a)):
            preds = np.array(bagged_predictions[idx])
            mean_pred = np.mean(preds)
            bias = (mean_pred - test_labels_numeric[idx]) ** 2
            variance = np.var(preds)
            biases_bagged.append(bias)
            variances_bagged.append(variance)

        avg_bias_bagged = np.mean(biases_bagged)
        avg_variance_bagged = np.mean(variances_bagged)
        mse_bagged = avg_bias_bagged + avg_variance_bagged

        print("Single Tree Result：")
        print(f"（Bias）：{avg_bias_single}")
        print(f"（Variance）：{avg_variance_single}")
        print(f"（MSE）：{mse_single}")

        print("\nResults of Bagging Model:")
        print(f"Average Bias: {avg_bias_bagged}")
        print(f"Average Variance: {avg_variance_bagged}")
        print(f"Mean Squared Error (MSE): {mse_bagged}")

        # Comparison of results
        print("\nComparison:")
        print(f"Bias Reduction: {avg_bias_single - avg_bias_bagged}")
        print(f"Variance Reduction: {avg_variance_single - avg_variance_bagged}")
        print(f"MSE Reduction: {mse_single - mse_bagged}")

