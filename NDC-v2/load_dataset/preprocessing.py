import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir, no_label='_no_label', print_stats=True):
    """Returns NDC data from given directory as a pandas dataframe, 
    dropping all unlabeled elements.

    Args:
        data_dir (str): the relative path of the NDC data CSV file
        no_label (str, optional): label associated with unlabeled text. 
            Defaults to '_no_label'.
        print_stats (bool, optional): option to print statistics of NDC data. 
            Defaults to True.

    Returns:
        pd.Dataframe: Dataframe containing NDC data
    """    
    df = pd.read_csv(data_dir)
    df.drop(df[df['label'] == no_label].index, inplace=True)

    if print_stats:
        print('Labels:', np.unique(df['label']))
        print('Countries:', np.unique(df['iso']))
        print('# of countries:', len(np.unique(df['iso'])))

    return df


def load_data_split(data_dir, print_stats=True):
    """Reads pre-split NDC data as a pandas dataframe, returning it as well as
    a tuple containing the text and labels of the training, testing, and
    validation sets.

    Args:
        data_dir (str): the relative path of the NDC data CSV file
        print_stats (bool, optional): option to print statistics of NDC data. 
            Defaults to True.

    Returns:
        Tuple[pd.Dataframe, Tuple[6 * (list[string],)]]: First element is raw
            data, second element is tuple containing 6 lists of strings: training 
            text and labels, testing text and labels, validation text and labels
    """    
    df = pd.read_csv(data_dir)

    train_text = []
    train_labels = []
    val_text = []
    val_labels = []
    test_text = []
    test_labels = []
  
    for _, row in df.iterrows():
        if row['split'] == 'train':
            train_text.append(row['sentence'])
            train_labels.append(row['label'])
        elif row['split'] == 'test':
            test_text.append(row['sentence'])
            test_labels.append(row['label'])
        elif row['split'] == 'validate':
            val_text.append(row['sentence'])
            val_labels.append(row['label'])

    # TODO: Move to unit tests file
    assert len(train_text) == len(train_labels)
    assert len(test_text)  == len(test_labels)
    assert len(val_text)   == len(val_labels)
    assert len(df)         == (len(train_text) + len(test_text) + len(val_text))

    # Print sentence statistics
    if print_stats:
        print(f'Training dataset contains {len(train_text)} sentences,',
              f'validation dataset contains {len(val_text)} sentences,',
              f'and test dataset contains {len(test_text)} sentences,',
              f'for a total of {len(train_text) + len(val_text) + len(test_text)}'
               ' sentences.')
  
    train_text = np.array(train_text)
    train_labels = np.array(train_labels)
    test_text = np.array(test_text)
    test_labels = np.array(test_labels)
    val_text = np.array(val_text)
    val_labels = np.array(val_labels)

    return df, (train_text, train_labels, test_text, test_labels, val_text, val_labels)


def split_train_test(df, df_test=None, print_stats=True):
    """Randomly splits NDC data into training, validation, and testing sets.

    Args:
        df (pd.Dataframe): Raw NDC data
        df_test ([type], optional): TODO (ask Eric?). Defaults to None.
        print_stats (bool, optional): option to print country and sentence
            statistics. Defaults to True.

    Returns:
        Tuple[6 * (list[str],)]: the training text and labels, testing text
            and labels, and validation text and labels
    """    
    if df_test is not None:
        blacklist = df_test.row_id.unique()
        for row_id in blacklist:
            df = df[df['row_id'] != row_id]
        df_by_iso = list(df.groupby('iso'))
        
    train, test = train_test_split(df_by_iso, train_size=0.80)
    test, validation = train_test_split(test, test_size=0.5)

    if print_stats:
        print(f'Training dataset contains {len(train)} countries,',
              f'test dataset contains {len(test)} countries,'
              f'and validation dataset contains {len(validation)} countries,',
              f'for a total of {len(train) + len(test) + len(validation)}'
               ' countries.')

    # Merge countries back into train and test datasets.
    # Each element of train/test is a tuple (iso, data), 
    # so grab data and respective column for each country.
    train_text   = np.concatenate([np.array(train[i][1]['sentence'])      
                   for i in range(len(train))])
    train_labels = np.concatenate([np.array(train[i][1]['label']) 
                   for i in range(len(train))])
    test_text    = np.concatenate([np.array(test[i][1]['sentence'])       
                   for i in range(len(test))])
    test_labels  = np.concatenate([np.array(test[i][1]['label'])          
                   for i in range(len(test))])
    val_text     = np.concatenate([np.array(validation[i][1]['sentence']) 
                   for i in range(len(validation))])
    val_labels   = np.concatenate([np.array(validation[i][1]['label'])    
                   for i in range(len(validation))])

    if print_stats:
        print(f'Training dataset contains {len(train_text)} sentences,',
              f'validation dataset contains {len(val_text)} sentences,',
              f'and test dataset contains {len(test_text)} sentences,',
              f'for a total of {len(train_text) + len(val_text) + len(test_text)}'
               ' sentences.')

    # TODO: Move to unit tests file 
    # Sanity check
    assert len(train_text) == len(train_labels)
    assert len(test_text)  == len(test_labels)
    assert len(val_text)   == len(val_labels)

    return (train_text, train_labels, 
            test_text, test_labels,
            val_text, val_labels)


def encode_labels(df, train_labels, val_labels, test_labels):
    """Encodes labels from strings to integers.

    Args:
        df (pd.Dataframe): Raw NDC data
        train_labels (list[str]): Labels from NDC training set
        val_labels (list[str]): Labels from NDC validation set
        test_labels (list[str]): Labels from NDC testing set

    Returns:
        Tuple[list[int], list[int], list[int], LabelEncoder]: Transformed
            training, validation, and testing set labels, and the encoder
            used in the transformation.
    """    
    encoder = LabelEncoder()
    encoder.fit(df['label'])

    train_labels = encoder.transform(train_labels)
    val_labels   = encoder.transform(val_labels)
    test_labels  = encoder.transform(test_labels)

    return (train_labels, val_labels, test_labels, encoder)
