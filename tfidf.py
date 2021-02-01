import os  # for accessing files
import re  # for tokenizing file names
import pandas as pd  # for data tools
from sklearn.feature_extraction.text import TfidfVectorizer  # for built-in tf-idf


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
        # Filter out non-text files and non-English documents
        if filename.endswith('.txt') and ('EN' in filename_tokens or 'EN_TR' in filename_tokens):
            # Get filename without extension for labelling purposes
            key = os.path.splitext(os.path.basename(filename))[0]
            keys.append(key)

            # Read in file
            documents.append(open(ndc_txt_dir + filename, encoding='utf-8').read())

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


tfidf_data = calc_tfidf('txt/', (1, 2))  # Calculate tf-idf on monograms

# Example usage
# ngram_ranking = rank_ngrams(tfidf_data, 10, 'AFG-INDC-EN')  # Find 10 highest tf-idf scoring words for Afghanistan's INDC
# print(ngram_ranking)

# tfidf_data.to_csv('tfidf.csv')  # Only run to output whole dataset

'''
Notes: (last updated 01/31/2021)
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
'''
