import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

data_dir = '20210302_NDC_Labeled_Sentences_V1/20210302_15_NDC_Labeled_Sentences_10_Words.csv'
data_dir_human = '20210302_NDC_Labeled_Sentences_V1/20210502_NDC_Labeled_Sentences_10_Words_with_Human_Labels.csv'


labels = ['adaptation',
           'agriculture',
           'economic',
           'energy',
           'environment',
           'equity',
           'industry',
           'land_use',
           'mitigation',
           'strategy',
           'waste']


def load_data(data_dir):
    df = pd.read_csv(data_dir)
    df.drop(df[df['label'] == '_no_label'].index, inplace=True)   # drop unlabeled sentences

    # print('Labels:', np.unique(df['label']))
    # print('Countries:', np.unique(df['iso']))
    # print('# of countries:', len(np.unique(df['iso'])))

    return df


def random_labels(test_text):
    predictions = []

    for sentence in test_text:
        predictions.append(labels[np.random.randint(11)])

    return predictions


def all_strategy_labels(test_text):
    predictions = []

    for sentence in test_text:
        predictions.append('strategy')

    return predictions


def simple_contains(test_text):
    predictions = []

    for sentence in test_text:
        for label in labels:
            if label in sentence:
                predictions.append(label)
                break
        else:
            predictions.append('_no_label')

    return predictions


def check_human_labels(df, label_human):
    true_labels = [df.iloc[i]['label'] for i in range(len(df))]
    human_labels = [df.iloc[i][label_human] for i in range(len(df))]
    return f1_score(true_labels, human_labels, average="macro")


if __name__ == '__main__':
    df = load_data(data_dir)
    text = [df.iloc[i]['sentence'] for i in range(len(df))]
    true_labels = [df.iloc[i]['label'] for i in range(len(df))]
    baselines = [random_labels, all_strategy_labels, simple_contains]
    for baseline in baselines:
        print(f'{baseline.__name__}: {f1_score(true_labels, baseline(text), average="macro")}')
    df_human = load_data(data_dir_human)
    print(f'Mandeep: {check_human_labels(df_human, "label_mandeep")}')
    print(f'Ethan: {check_human_labels(df_human, "label_ethan")}')
