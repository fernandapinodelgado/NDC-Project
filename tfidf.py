import os  # for accessing files
import re  # for tokenizing file names
import pandas as pd  # for data tools
from sklearn.feature_extraction.text import TfidfVectorizer  # for built-in tf-idf

countries = []


def calc_tfidf(ndc_txt_dir: str, ngram_range: tuple):
    """
    Calculates tf-idf scores for every document-word pair in the directory.
    :param ndc_txt_dir: path to directory with ndc txt data
    :param ngram_range: range of n-grams to use in calculation
    :return: n x m DataFrame: n = # of documents, m = # of unique n-grams
    """
    documents = []
    keys = []

    # Load text data from files
    for filename in os.listdir(ndc_txt_dir):
        # Split filename by '-' or '.' in order to find document language token (e.g. 'EN', 'ES', 'FR')
        filename_tokens = re.split(r'[-|.]', filename)
        country = filename_tokens[0]
        type = filename_tokens[1]
        lang = filename_tokens[2]

        # Skip copies of EU INDCs
        if len(filename_tokens) == 5 and filename_tokens[3] == 'EU28':
            continue

        # Filter out non-text files and non-English documents
        if filename.endswith('.txt') and (lang == 'EN' or lang == 'EN_TR'):
            # Check if document for country already exists
            if country in keys and type == 'NDC':
                idx = keys.index(country)
                documents[idx] = open(ndc_txt_dir + filename, encoding='utf-8').read()
            elif country not in keys:
                # Get filename without extension for labelling purposes
                keys.append(country)
                # Read in file
                documents.append(open(ndc_txt_dir + filename, encoding='utf-8').read())

    global countries
    countries = keys

    # Calculate tf-idf scores for each document-word pair
    # Source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df.index = keys
    return df


def rank_ngrams(df: pd.DataFrame, count, query, descending=True):
    """
    Returns a ranking of words with the highest/lowest tf-idf scores for a given document.
    :param df: DataFrame with tf-idf data
    :param count: number of words to output
    :param query: document to query words for
    :param descending: Toggles descending/ascending order, True by default
    :return: Series containing words and tf-idf scores
    """
    df = df.transpose()
    if descending:
        return df.nlargest(count, query)[query]
    else:
        return df.nsmallest(count, query)[query]


def rank_docs(df: pd.DataFrame, count, query, descending=True):
    """
    Returns a ranking of documents with the highest/lowest tf-idf scores for a given n-gram.
    :param df: DataFrame with tf-idf data
    :param count: number of documents to output
    :param query: word to query documents for
    :param descending: Toggles descending/ascending order, True by default
    :return: Series containing documents and tf-df scores
    """
    if descending:
        return df.nlargest(count, query)[query]
    else:
        return df.nsmallest(count, query)[query]


tfidf_data = calc_tfidf('txt/', (1, 1))  # Calculate tf-idf on monograms

# Example usage
# ngram_ranking = rank_ngrams(tfidf_data, 10, 'EU28')  # Find 10 highest tf-idf scoring words for Afghanistan's INDC
# # print(ngram_ranking)
# #
# # ranking = rank_docs(tfidf_data, 10, 'wildfires')
# # print(ranking)

# tfidf_data.to_csv('tfidf.csv')  # Only run to output whole dataset

monograms = {}
for country in countries:
    monogram_ranking = rank_ngrams(tfidf_data, 20, country)
    monograms[country] = list(monogram_ranking.index.values)

df_monograms = pd.DataFrame(monograms)
df_monograms = df_monograms.transpose()
df_monograms.to_csv('monograms.csv', header=None)

tfidf_data = calc_tfidf('txt/', (2, 2))  # Calculate tf-idf on bigrams

bigrams = {}
for country in countries:
    bigram_ranking = rank_ngrams(tfidf_data, 20, country)
    bigrams[country] = list(bigram_ranking.index.values)

df_bigrams = pd.DataFrame(bigrams)
df_bigrams = df_bigrams.transpose()
df_bigrams.to_csv('bigrams.csv', header=None)

tfidf_data = calc_tfidf('txt/', (3, 3))  # Calculate tf-idf on trigrams

trigrams = {}
for country in countries:
    trigram_ranking = rank_ngrams(tfidf_data, 20, country)
    trigrams[country] = list(trigram_ranking.index.values)

df_trigrams = pd.DataFrame(trigrams)
df_trigrams = df_trigrams.transpose()
df_trigrams.to_csv('trigrams.csv', header=None)


'''
Notes: (last updated 02/04/2021)
- Filter out country names and demonyms in word ranking
- Expected common words like "climate" and "change" end up with high tf-idf scores. Why?
  I expect that those common words should have low scores across all documents since they
  should be found in every document in the corpus. Perhaps misunderstanding tf-idf or
  something interesting with the structure of the documents.
- Use some unsupervised clustering method to find similar groups of documents
- Implement method to save tf-idf matrix so we don't have to re-calculate it every time,
  currently very time consuming for bigram and trigram queries
- Is there any difference between INDCs and NDCs?
- rank_ngrams and rank_docs are really similar functions, maybe try to merge them into one
- Some countries only have NDCs in a non-English language, like France.
'''
