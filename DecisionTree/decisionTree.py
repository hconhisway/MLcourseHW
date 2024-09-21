import csv
import math
from collections import Counter, defaultdict


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


class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=6):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, attributes):
        self.tree = self._build_tree(data, attributes, depth=0)

    def _best_attribute(self, data, attributes):
        base_criterion_value = self._criterion_value([row[-1] for row in data])
        best_gain = -1
        best_attr = None
        for attr in attributes:
            subsets = defaultdict(list)
            for row in data:
                subsets[row[attr]].append(row)
            weighted_criterion = 0
            for subset in subsets.values():
                weighted_criterion += (len(subset) / len(data)) * self._criterion_value([row[-1] for row in subset])
            gain = base_criterion_value - weighted_criterion
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr

    def _criterion_value(self, labels):
        if self.criterion == 'entropy':
            return entropy(labels)
        elif self.criterion == 'majority_error':
            return majority_error(labels)
        elif self.criterion == 'gini_index':
            return gini_index(labels)

    def _build_tree(self, data, attributes, depth):
        labels = [row[-1] for row in data]
        if len(set(labels)) == 1:
            return DecisionTreeNode(is_leaf=True, prediction=labels[0])
        if not attributes or depth == self.max_depth:
            majority_label = Counter(labels).most_common(1)[0][0]
            return DecisionTreeNode(is_leaf=True, prediction=majority_label)
        best_attr = self._best_attribute(data, attributes)
        if best_attr is None:
            majority_label = Counter(labels).most_common(1)[0][0]
            return DecisionTreeNode(is_leaf=True, prediction=majority_label)
        node = DecisionTreeNode(attribute=best_attr)
        attr_values = set(row[best_attr] for row in data)
        for value in attr_values:
            subset = [row for row in data if row[best_attr] == value]
            if not subset:
                majority_label = Counter(labels).most_common(1)[0][0]
                node.children[value] = DecisionTreeNode(is_leaf=True, prediction=majority_label)
            else:
                new_attrs = attributes.copy()
                new_attrs.remove(best_attr)
                node.children[value] = self._build_tree(subset, new_attrs, depth + 1)
        return node

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
            # Handle missing values
            for idx in range(len(row)-1):  # Exclude label
                if row[idx] == 'unknown':
                    if handle_unknown == 'value':
                        pass  # Keep 'unknown' as a separate value
                    elif handle_unknown == 'missing':
                        if idx in numerical_attrs:
                            # Replace 'unknown' with median
                            row[idx] = medians[idx]
                        else:
                            # Replace 'unknown' with majority value
                            row[idx] = majority_values[idx]
                else:
                    if idx in numerical_attrs:
                        row[idx] = float(row[idx])
            # Binarize numerical attributes
            for idx in numerical_attrs:
                if row[idx] == 'unknown':
                    # If still 'unknown' (when handle_unknown == 'value'), set to median
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


if __name__ == "__main__":
    import pandas as pd

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
    results_a = []
    train_data_a = load_data('bank/train.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                             handle_unknown='value')
    test_data_a = load_data('bank/test.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                            handle_unknown='value')

    criteria = ['entropy', 'majority_error', 'gini_index']
    max_depths = list(range(1, 17))  # 1 to 16

    for criterion in criteria:
        for depth in max_depths:
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(train_data_a, attribute_indices.copy())

            train_error = tree.evaluate(train_data_a)
            test_error = tree.evaluate(test_data_a)

            results_a.append({
                'Criterion': criterion,
                'Max Depth': depth,
                'Training Error': train_error,
                'Test Error': test_error
            })
            print(
                f'Criterion: {criterion}, Max Depth: {depth}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}')

    # Convert results to DataFrame and display
    results_df_a = pd.DataFrame(results_a)
    print('\nExperiment (a) Results Summary:')
    print(results_df_a)

    # Experiment (b): Replace 'unknown' with majority value
    print("\nExperiment (b): Replace 'unknown' with majority value")
    results_b = []

    # Load and preprocess data for experiment (b)
    train_data_b = load_data('bank/train.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                             handle_unknown='missing')
    test_data_b = load_data('bank/test.csv', numerical_attrs, categorical_attrs, medians, majority_values,
                            handle_unknown='missing')

    for criterion in criteria:
        for depth in max_depths:
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(train_data_b, attribute_indices.copy())

            train_error = tree.evaluate(train_data_b)
            test_error = tree.evaluate(test_data_b)

            results_b.append({
                'Criterion': criterion,
                'Max Depth': depth,
                'Training Error': train_error,
                'Test Error': test_error
            })
            print(
                f'Criterion: {criterion}, Max Depth: {depth}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}')

    # Convert results to DataFrame and display
    results_df_b = pd.DataFrame(results_b)
    print('\nExperiment (b) Results Summary:')
    print(results_df_b)

    # # train_data = load_data('bank/train.csv')
    # # test_data = load_data('bank/test.csv')
    #
    # criteria = ['entropy', 'majority_error', 'gini_index']
    # max_depths = [1, 2, 3, 4, 5, 6]
    #
    # results = []
    #
    # for criterion in criteria:
    #     for depth in max_depths:
    #         tree = DecisionTree(criterion=criterion, max_depth=depth)
    #         tree.fit(train_data, attribute_indices.copy())
    #
    #         train_error = tree.evaluate(train_data)
    #         test_error = tree.evaluate(test_data)
    #
    #         results.append({
    #             'Criterion': criterion,
    #             'Max Depth': depth,
    #             'Training Error': train_error,
    #             'Test Error': test_error
    #         })
    #         print(
    #             f'Criterion: {criterion}, Max Depth: {depth}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}')
    #
    # # Convert results to DataFrame and display
    # results_df = pd.DataFrame(results)
    # print('\nResults Summary:')
    # print(results_df)
