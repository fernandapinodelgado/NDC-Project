import random
import torch
from transformers import BertTokenizer


def tokenize(text, bert_version, max_len=400):
    """Tokenizes sequences for input to BERT, truncating each one to max_len.

    Args:
        text (List[str]): list of samples from NDC dataset
        bert_version (str): version of BERT to be used for tokenizing
        max_len (int): length to truncate each sequence to. Defaults to 400.

    Returns:
        Tuple[List[List[int]], BertTokenizer]: a tuple containing the list of 
            tokenized inputs as well as the tokenizer.
    """    
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(bert_version, do_lower_case=True)

    full_input_ids = []

    # Tokenize all training examples
    print('Tokenizing {:,} training samples...'.format(len(text)))

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(text), 
                                           num_desired_updates=10)

    # For each training example...
    for sample in text:
        
        # Report progress.
        if ((len(full_input_ids) % update_interval) == 0):
            print('  Tokenized {:,} samples.'.format(len(full_input_ids)))

        # Tokenize the sentence.
        input_ids = tokenizer.encode(text=sample,            
                                    add_special_tokens=True, 
                                    max_length=max_len,
                                    truncation=True,    
                                    padding=False)     
                                    
        # Add the tokenized result to our list.
        full_input_ids.append(input_ids)
        
    print('DONE.')
    print('{:>10,} samples'.format(len(full_input_ids)))

    return (full_input_ids, tokenizer)


def sort_data(full_input_ids, labels):
    """Sort the two lists together by the length of the input sequence.

    Args:
        full_input_ids (List[List[int]]): list of tokenized inputs for BERT
        train_labels (List[str]): list of data labels

    Returns:
        List[Tuple[List[int], str]]: list of tuples of each tokenized input 
            and its label.
    """    
    samples = sorted(zip(full_input_ids, labels), key=lambda x: len(x[0]))
    print('Shortest sample:', len(samples[0][0]))
    print('Longest sample:', len(samples[-1][0]))

    return samples


def select_batches(samples, batch_size):
    """Constructs smart batches of size batch_size from processed samples.
    This involves:
        1. Picking a random starting point in the sorted list of samples.
        2. Grabbing a contiguous batch of samples starting from that point.
        3. Deleting those samples from the list, and repeating until all of
            the samples have been grabbed.
    This results in some fragmentation of the list, which means it won't be as
    efficient as simply slicing the list into consecutive batches in sorted 
    order. The benefit is that our path through the training set can still 
    have a degree of randomness.

    Args:
        samples (List[Tuple[List[int], str]]): list of tuples of each 
            tokenized input and its label
        batch_size (int): the size of each batch

    Returns:
        Tuple[List[List[List[int]]], List[List[str]]]: the sample batches and 
            label batches
    """
    # TODO: batch_size = wandb.config.batch_size
    # List of batches that we'll construct.
    batch_ordered_sentences = []
    batch_ordered_labels = []

    print('Creating training batches of size {:}'.format(batch_size))

    # Loop over all of the input samples...    
    while len(samples) > 0:
        
        # Report progress.
        if ((len(batch_ordered_sentences) % 500) == 0):
            print('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))

        # `to_take` is our actual batch size. It will be `batch_size` until 
        # we get to the last batch, which may be smaller. 
        to_take = min(batch_size, len(samples))

        # Pick a random index in the list of remaining samples to start
        # our batch at.
        select = random.randint(0, len(samples) - to_take)

        # Select a contiguous batch of samples starting at `select`.
        batch = samples[select:(select + to_take)]

        # Each sample is a tuple--split them apart to create a separate list of 
        # sequences and a list of labels for this batch.
        batch_ordered_sentences.append([s[0] for s in batch])
        batch_ordered_labels.append([s[1] for s in batch])

        # Remove these samples from the list.
        del samples[select:select + to_take]

    print('\n  DONE - {:,} batches.'.format(len(batch_ordered_sentences)))

    return (batch_ordered_sentences, batch_ordered_labels)


def add_padding(batch_ordered_sentences, batch_ordered_labels, tokenizer):
    """Pads batches to a uniform length.

    Args:
        batch_ordered_sentences (List[List[List[int]]]): batches of samples
        batch_ordered_labels (List[List[str]]]): batches of labels
        tokenizer (BertTokenizer): the tokenizer used to transform the input

    Returns:
        Tuple[3 * (torch.Tensor,)]: a tuple containing the padded batches, the
            attention masks, and the inputs' labels, all as PyTorch Tensors.
    """    
    py_inputs = []
    py_attn_masks = []
    py_labels = []

    # For each batch...
    for (batch_inputs, batch_labels) in zip(batch_ordered_sentences, batch_ordered_labels):

        # New version of the batch, this time with padded sequences and now with
        # attention masks defined.
        batch_padded_inputs = []
        batch_attn_masks = []
        
        # First, find the longest sample in the batch. 
        # Note that the sequences do currently include the special tokens!
        max_size = max([len(sen) for sen in batch_inputs])

        # For each input in this batch...
        for sen in batch_inputs:
            
            # How many pad tokens do we need to add?
            num_pads = max_size - len(sen)

            # Add `num_pads` padding tokens to the end of the sequence.
            padded_input = sen + [tokenizer.pad_token_id]*num_pads

            # Define the attention mask--it's just a `1` for every real token
            # and a `0` for every padding token.
            attn_mask = [1] * len(sen) + [0] * num_pads

            # Add the padded results to the batch.
            batch_padded_inputs.append(padded_input)
            batch_attn_masks.append(attn_mask)

        # Our batch has been padded, so we need to save this updated batch.
        # We also need the inputs to be PyTorch tensors, so we'll do that here.
        py_inputs.append(torch.tensor(batch_padded_inputs))
        py_attn_masks.append(torch.tensor(batch_attn_masks))
        py_labels.append(torch.tensor(batch_labels))

    return (py_inputs, py_attn_masks, py_labels)
