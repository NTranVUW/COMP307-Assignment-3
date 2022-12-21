import sys
import pandas as pd


def test_naive_bayes(data, probability_table, class_names):
    predictions = []
    predictions_prob = []
    correct = 0
    for index, row in data.iterrows():
        pred_class, pred_prob = predict(row, probability_table, class_names)
        predictions.append(pred_class)
        predictions_prob.append(pred_prob)
        if pred_class == row['class']:
            correct += 1
    accuracy = correct / len(data)
    predictions_df = pd.DataFrame(
        {'class': data['class'], 'prediction': predictions, 'prediction_prob': predictions_prob})
    print(predictions_df)
    print("\nTest Accuracy: {}%".format(accuracy * 100))


def predict(data, probability_table, class_names):
    pred_class = None
    score = {}
    for class_name in class_names:
        score[class_name] = probability_table[class_name]
        for feature in data.keys()[1:]:
            score[class_name] *= probability_table[(class_name, feature, data[feature])]
    pred_class = max(score, key=score.get)
    pred_prob = score[pred_class]
    return pred_class, pred_prob


def get_probability_table(frequency_table, data, class_names):
    P = {}
    # tables = []
    for class_name in class_names:
        P[class_name] = frequency_table[class_name]['total'] / data.shape[0]
        # table = []
        # table.append(P[class_name])
        for feature in data.columns[1:]:
            for value in data[feature].unique():
                P[(class_name, feature, value)] = frequency_table[class_name][(feature, value)] / \
                                                  frequency_table[class_name]['total']
                # table.append(P[(class_name, feature, value)])
        # tables.append(table)
    # feature_names = []
    # for feature in data.columns[1:]:
    # for value in data[feature].unique():
    # feature_names.append("P({} = {}|Y=y)".format(feature, value))
    return P


def get_frequency_table(data, class_names):
    tables = {}
    for class_name in class_names:
        table = {}
        table['total'] = data[data['class'] == class_name].shape[0] + 1
        for feature in data.columns[1:]:
            for value in data[feature].unique():
                table[(feature, value)] = data[(data[feature] == value) & (data['class'] == class_name)].shape[0] + 1
        tables[class_name] = table
    return tables


def train_naive_bayes(data, class_names):
    frequency_table = get_frequency_table(data, class_names)
    return get_probability_table(frequency_table, data, class_names)


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    # print("Usage: python naive_bayes.py breast-cancer-training.csv breast-cancer-test.csv")
    # exit(1)
    # train_data = pd.read_csv(sys.argv[1], index_col=0)
    # test_data = pd.read_csv(sys.argv[2], index_col=0)

    train_data = pd.read_csv('breast-cancer-training.csv', index_col=0)
    test_data = pd.read_csv('breast-cancer-test.csv', index_col=0)

    class_names = train_data['class'].unique()

    probability_table = train_naive_bayes(train_data, class_names)

    test_naive_bayes(test_data, probability_table, class_names)
