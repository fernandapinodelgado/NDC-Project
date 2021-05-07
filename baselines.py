import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

data_dir = 'csv/20210302_15_NDC_Labeled_Sentences_10_Words.csv'
data_dir_human = 'csv/20210502_NDC_Labeled_Sentences_10_Words_with_Human_Labels.csv'

words_dir = 'csv/20210302_03_words_in_subject_labels.csv'


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


def load_label_words(words_dir):
    df_words = pd.read_csv(words_dir)
    temp_dict = df_words.to_dict('index')
    word_dict = {}
    for entry in temp_dict.values():
        words = entry['subject_words'].split(',')
        words = [s.strip() for s in words]
        word_dict[entry['subject_label']] = words
    return word_dict


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


def check_contains(sentence, label_words):
    for label in labels:
        words = label_words[label]
        for word in words:
            if word in sentence.lower():
                return label
    else:
        return '_no_label'


def simple_contains(test_text):
    predictions = []
    label_words = load_label_words(words_dir)

    for sentence in test_text:
        predictions.append(check_contains(sentence, label_words))

    return predictions


def check_human_labels(df, label_human, name, dropna=False):
    if dropna:
        df.dropna(inplace=True)
    true_labels = [df.iloc[i]['label'] for i in range(len(df))]
    human_labels = [df.iloc[i][label_human] for i in range(len(df))]
    print(f'{name} F1: {f1_score(true_labels, human_labels, average="macro")}')
    print(f'{name} accuracy: {accuracy_score(true_labels, human_labels)}')


if __name__ == '__main__':
    df = load_data(data_dir)
    text = [df.iloc[i]['sentence'] for i in range(len(df))]
    true_labels = [df.iloc[i]['label'] for i in range(len(df))]
    baselines = [random_labels, all_strategy_labels, simple_contains]
    for baseline in baselines:
        print(f'{baseline.__name__} F1: {f1_score(true_labels, baseline(text), average="macro")}')
        print(f'{baseline.__name__} accuracy: {accuracy_score(true_labels, baseline(text))}')
    df_human = load_data(data_dir_human)
    text = [df_human.iloc[i]['sentence'] for i in range(len(df_human))]
    df_human['random'] = random_labels(text)
    df_human['contains'] = simple_contains(text)
    df_human.to_csv('csv/baseline_labels.csv')
    check_human_labels(df_human, "label_mandeep", 'Mandeep')
    check_human_labels(df_human, "label_ethan", 'Ethan', dropna=True)

