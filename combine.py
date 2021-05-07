import pandas as pd

data_human = 'csv/20210502_NDC_Labeled_Sentences_10_Words_with_Human_Labels.csv'
data_untrained = 'csv/untrained_BERT_labels.csv'
data_trained = 'csv/trained_BERT_labels.csv'
data_baseline = 'csv/baseline_labels.csv'

df = pd.read_csv(data_human)
df.drop(df[df['label'] == '_no_label'].index, inplace=True)   # drop unlabeled sentences
df_untrained = pd.read_csv(data_untrained)
df_trained = pd.read_csv(data_trained)
df_baseline = pd.read_csv(data_baseline)

df['random'] = list(df_baseline['random'])
df['contains'] = list(df_baseline['contains'])
df['untrained'] = list(df_untrained['untrained_BERT'])
df['trained'] = list(df_trained['trained_BERT'])

print(len(df))
print(len(df_baseline['random']))
print(len(df_baseline['contains']))
print(len(df_untrained['untrained_BERT']))
print(len(df_trained['trained_BERT']))

print(df)

df.to_csv('csv/20210502_NDC_Labeled_Sentences_10_Words_with_Human_Labels_and_Results.csv', index=False)
