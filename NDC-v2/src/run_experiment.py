import numpy as np
import os
import torch
import wandb

import sys
sys.path.insert(0, '../')

from models.train import eval_model, print_accuracy, train_loop
from models.load_pretrained import get_optimizer_scheduler, load_model
from load_dataset.preprocessing import encode_labels, load_data, load_data_split, split_train_test
from load_dataset.batching import add_padding, make_smart_batches, select_batches, sort_data, tokenize

wandb.login()
os.system("env WANDB_PROJECT=ndc-project")
os.system("env WANDB_LOG_MODEL=true")

lr = 7.5e-5
batch_size = 8
epochs = 10
warmup_steps = 500
wandb_config = {
    'learning_rate': lr,
    'batch_size': batch_size,
    'epochs': epochs,
    'warmup_steps': warmup_steps
}
wandb.init(
    project='ndc-project',
    config=wandb_config, 
    name=f'bert-{lr}-lr_{batch_size}-batch_'
        f'{epochs}-epochs_{warmup_steps}-warmups'
)


def standard_run(data_dir, bert_version):
    # PREPROCESSING
    df = load_data(data_dir)
    train_text, train_labels, test_text, test_labels, val_text, val_labels \
                                                        = split_train_test(df)
    # see_train_dist(train_labels)
    train_labels, val_labels, test_labels, encoder = encode_labels(
        train_labels, val_labels, test_labels
    )

    # BATCHING
    py_inputs, py_attn_masks, py_labels = make_smart_batches(
        train_text, train_labels, batch_size
    )

    # LOAD_PRETRAINED
    model, device = load_model(bert_version)
    optimizer, scheduler = get_optimizer_scheduler(model, py_inputs, wandb)

    # TRAINING
    train_loop(
        model, py_inputs, py_attn_masks, py_labels, epochs, 
        train_text, train_labels, val_text, val_labels, batch_size, 
        device, optimizer, scheduler, encoder, wandb
    )

    # EVALUATION
    # clear and load best model
    model.load_state_dict(torch.load("best_model.pt"))
    model.to(device)
    predictions, true_labels, _, _, _ = eval_model(
        model, test_text, test_labels, batch_size, device
    )
    test_acc, f1_macro = print_accuracy(predictions, true_labels, encoder, wandb)
    print('Best Test Accuracy: {:.3f}'.format(test_acc))
    print('Best Test Macro-F1: {:.3f}'.format(f1_macro))


def label_set(row, train, val, test):
    if row['sentence'] in train:
        return 'train'
    elif row['sentence'] in val:
        return 'val'
    elif row['sentence'] in test:
        return 'test'
    else:
        return 'na'


def save_set_split(df, train, val, test):
    df_saved = df.copy()
    df_saved['set'] = df.apply(lambda row: label_set(row, train, val, test), axis=1)
    return df_saved


def save_model_preds(df, encoder, model, batch_size, device, col_name):
    all_text = df['sentence'].to_numpy()
    all_labels = df['label'].to_numpy()
    encoded_labels = encoder.transform(all_labels)
    preds = []
    for i in range(len(all_text)):
        prediction, true_label, _, _, _ = eval_model(model, all_text[i:i+1], encoded_labels[i:i+1], batch_size, device)
        prediction = np.concatenate(prediction, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        pred = np.argmax(prediction, axis=1).flatten()
        preds.append(pred[0])
    df_saved = df.copy()
    df_saved[col_name] = encoder.inverse_transform(preds)
    return df_saved


if __name__ == "__main__":
    data_dir='../20210531_new_bert.csv'
    bert_version='bert-base-uncased'
    standard_run(data_dir, bert_version)
    # df, data_split = load_data_split(data_dir)
    # train_text, train_labels, test_text, test_labels, val_text, val_labels = data_split
    # train_labels, val_labels, test_labels, encoder = encode_labels(df, train_labels, val_labels, test_labels)
    # full_input_ids, tokenizer = tokenize(train_text, bert_version)
    # train_samples = sort_data(full_input_ids, train_labels)
    # batch_ordered_sentences, batch_ordered_labels = select_batches(train_samples, batch_size)
    # py_inputs, py_attn_masks, py_labels = add_padding(batch_ordered_sentences, batch_ordered_labels, tokenizer)
    
    # model, device = load_model(bert_version)
    # optimizer, scheduler = get_optimizer_scheduler(model, py_inputs, wandb)
    # train_loop(model, py_inputs, py_attn_masks, py_labels, epochs, train_text, val_text, train_labels, val_labels, batch_size, device, optimizer, scheduler, encoder, wandb)  
