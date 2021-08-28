import argparse
import numpy as np
import os
import torch
import wandb

import sys
sys.path.insert(0, '../')

from models.train import eval_model, print_accuracy, train_loop
from models.load_pretrained import get_optimizer_scheduler, load_model
from load_dataset.preprocessing import encode_labels, load_data, load_data_split, split_train_test
from load_dataset.batching import make_smart_batches


parser = argparse.ArgumentParser(
    description='Runs a standard NDC BERT experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'data_dir', type=str,
    help='the filepath of the NDC data .csv file'
)
parser.add_argument(
    '--lr', type=float, default=7.5e-5, help='learning rate for Adam optimizer'
)
parser.add_argument(
    '--eps', type=float, default=1e-8, help='epsilon for Adam optimizer'
)
parser.add_argument(
    '--batch', type=int, default=8, help='batch size for BERT training'
)
parser.add_argument(
    '--epochs', type=int, default=10, help='number of epochs to train for'
)
parser.add_argument(
    '--warmup', type=int, default=500, help='number of warmup steps'
)
parser.add_argument(
    '--pre_split', action='store_true',
    help='option to use pre-split data rather than new random'
         ' train-test-validation split'
)
parser.add_argument(
    '--bert-version', type=str, default='bert-base-uncased',
    help='version of BERT from huggingface.co/models'
)

args = parser.parse_args()


wandb.login()
os.system("env WANDB_PROJECT=ndc-project")
os.system("env WANDB_LOG_MODEL=true")

wandb_config = {
    'learning_rate': args.lr,
    'epsilon':       args.eps,
    'batch_size':    args.batch,
    'epochs':        args.epochs,
    'warmup_steps':  args.warmup,
    'bert_version':  args.bert_version,
    'pre_split':     args.pre_split
}
wandb.init(
    project='ndc-project',
    config=wandb_config, 
    name=f'{wandb_config.bert_version}_'
         f'{wandb_config.lr}-lr_'
         f'{wandb_config.eps}-eps_'
         f'{wandb_config.batch_size}-batch_'
         f'{wandb_config.epochs}-epochs_'
         f'{wandb_config.warmup_steps}-warmups_'
         + 'pre-split' if args.pre_split else ''
)


def standard_run():
    # PREPROCESSING
    if args.pre_split:
        df, data_split = load_data_split(args.data_dir)
        train_text, train_labels, test_text, test_labels, val_text, val_labels \
            = data_split
    else:
        df = load_data(args.data_dir)
        train_text, train_labels, test_text, test_labels, val_text, val_labels \
            = split_train_test(df)

    train_labels, val_labels, test_labels, encoder = encode_labels(
        df, train_labels, val_labels, test_labels
    )

    # BATCHING
    py_inputs, py_attn_masks, py_labels = make_smart_batches(
        train_text, train_labels, args.batch
    )

    # LOAD_PRETRAINED
    model, device = load_model(args.bert_version)
    optimizer, scheduler = get_optimizer_scheduler(
        model, py_inputs, args.lr, args.eps, args.epochs, args.warmup
    )

    # TRAINING
    train_loop(
        model, py_inputs, py_attn_masks, py_labels, args.epochs, 
        train_text, train_labels, val_text, val_labels, args.batch_size, 
        device, optimizer, scheduler, encoder, wandb
    )

    # EVALUATION
    model.load_state_dict(torch.load("best_model.pt"))
    model.to(device)
    predictions, true_labels, _ = eval_model(
        model, test_text, test_labels, args.batch_size, device
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
        prediction, true_label, _ = eval_model(
            model, all_text[i:i+1], encoded_labels[i:i+1], batch_size, device
        )
        prediction = np.concatenate(prediction, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        pred = np.argmax(prediction, axis=1).flatten()
        preds.append(pred[0])
    df_saved = df.copy()
    df_saved[col_name] = encoder.inverse_transform(preds)
    return df_saved


if __name__ == "__main__":
    # data_dir='../20210531_new_bert.csv'
    standard_run()
