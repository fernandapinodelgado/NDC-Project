import numpy as np
import random
import time
import torch
from load_dataset.batching import make_smart_batches
from models.load_pretrained import get_learning_rate
from src.utils import good_update_interval, format_time

seed_val = 321

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def train_loop(model, py_inputs, py_attn_masks, py_labels, epochs, train_text,
        train_labels, val_text, val_labels, batch_size, device, optimizer,
        scheduler, encoder, wandb):
    """Handles the training and logging of the BERT model, with early stopping.

    Args:
        model (transformers.BertForSequenceClassification): 
        py_inputs ([type]): the padded batches
        py_attn_masks ([type]): the attention masks
        py_labels ([type]): the encoded labels
        epochs ([type]): number of epochs
        train_text ([type]): the training set
        val_text ([type]): the validation set
        train_labels ([type]): the training set's labels
        val_labels ([type]): the validation set's labels
        batch_size ([type]): the batch size used
        device ([type]): the GPU being used for computations
        optimizer ([type]): the optimizer being used
        scheduler ([type]): the scheduler being used
        encoder ([type]): the encoder used for the labels
        wandb ([type]): the wandb object
    """    
    training_stats = []

    update_interval = good_update_interval(total_iters=len(py_inputs), num_desired_updates=10)

    total_t0 = time.time()

    global_step = 0
    num_val_inc = 0
    prev_val_loss = 0
    min_val_loss = 0
    VAL_STOP = 3

    wandb.log({'learning_rate':get_learning_rate(optimizer)}, step=global_step)
    wandb.log({'training accuracy': 1/12}, step=global_step)
    wandb.log({'validation accuracy': 1/12}, step=global_step)

    for epoch_i in range(epochs):
        print(f"\n{'='*8} Epoch {epoch_i+1} / {epochs} {'='*8}")

        if epoch_i > 0:
            py_inputs, py_attn_masks, py_labels = make_smart_batches(train_text, train_labels, batch_size)
        
        print(f"Training on {len(py_inputs)} batches...")

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step in range(len(py_inputs)):
            global_step += 1

            # do_progress_update
            if step % update_interval == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                steps_per_sec = (time.time() - t0) / step
                remaining_sec = steps_per_sec * (len(py_inputs) - step)
                remaining = format_time(remaining_sec)

                print(f"    Batch {step} of {len(py_inputs)}.   Elapsed: {elapsed}.   Remaining: {remaining}")

            # send_batch_to_gpu
            b_input_ids = py_inputs[step].to(device)
            b_input_mask = py_attn_masks[step].to(device)
            b_labels = py_labels[step].to(device)

            model.zero_grad()
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            wandb.log({'train_batch_loss':outputs.loss.item()}, step=global_step)

            total_train_loss += outputs.loss

            outputs.loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            wandb.log({'learning_rate':get_learning_rate(optimizer)}, step=global_step)

            scheduler.step()

        avg_train_loss = total_train_loss / len(py_inputs)

        training_time = format_time(time.time() - t0)

        # eval loop
        train_preds, train_true_labels, _ = eval_model(model, train_text, train_labels, batch_size, device)
        val_preds, val_true_labels, val_loss = eval_model(model, val_text, val_labels, batch_size, device)

        # print val accuracy
        print_accuracy(train_preds, train_true_labels, encoder, wandb, step=global_step, title='training')
        val_acc, _ = print_accuracy(val_preds, val_true_labels, encoder, wandb, step=global_step, title='validation')

        wandb.log({'train_epoch_loss':avg_train_loss}, step=global_step)
        wandb.log({'validation_epoch_loss':val_loss}, step=global_step)

        print("")
        print("   Average training loss: {0:.2f}".format(avg_train_loss))
        print("   Average validation loss: {0:.2f}".format(val_loss))
        print(f"   Training epoch took: {training_time}")

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
                'Validation Loss': val_loss
            }
        )

        # early stopping
        if prev_val_loss == 0:
            pass
        elif prev_val_loss < val_loss:
            num_val_inc += 1
        else:
            num_val_inc = 0
            prev_val_loss = val_loss

        if min_val_loss == 0:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{val_loss}.pt')
        elif val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{val_loss}.pt')

        if num_val_inc >= VAL_STOP:
            print(f"Early stopping at epoch {epoch_i + 1}.")
            print("\nTraining complete!")
            print("Total training took {:} (h:mm:ss)\n".format(format_time(time.time()-total_t0)))
            return


def eval_model(model, test_text, test_labels, batch_size, device):
    # Prediction on test set
    # Refactor to eval_loop(model, data)

    py_inputs, py_attn_masks, py_labels = make_smart_batches(test_text, test_labels, batch_size)

    print('Predicting labels for {:,} test sentences...'.format(len(test_labels)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(py_inputs), num_desired_updates=10)

    # Measure elapsed time.
    t0 = time.time()

    total_loss = 0

    # For each batch of training data...
    for step in range(0, len(py_inputs)):

        # Progress update every 100 batches.
        if step % update_interval == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Calculate the time remaining based on our progress.
            steps_per_sec = (time.time() - t0) / step
            remaining_sec = steps_per_sec * (len(py_inputs) - step)
            remaining = format_time(remaining_sec)

            # Report progress.
            print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}.  Remaining: {:}'.format(step, len(py_inputs), elapsed, remaining))

        # Copy the batch to the GPU.
        b_input_ids = py_inputs[step].to(device)
        b_input_mask = py_attn_masks[step].to(device)
        b_labels = py_labels[step].to(device)
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1] # used to be 0, should be 1 or try .logits?
        total_loss += outputs.loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    avg_loss = total_loss / len(py_inputs)

    print('    DONE.')
    return (predictions, true_labels, avg_loss)


from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix


def print_accuracy(predictions, true_labels, encoder, wandb, step=None, title=None):
    # Combine the results across the batches.
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Choose the label with the highest score as our prediction.
    preds = np.argmax(predictions, axis=1).flatten()

    # Calculate simple flat accuracy -- number correct over total number.
    accuracy = (preds == true_labels).mean()
    # sample k errors to find pattern

    # use sklearn.metrics.f1_score here, get average f1 use macro, per class f1 pass in None to average
    # use sklearn.metrics.confusion_matrix here, plot into wandb
    # calculate f1 and confusion on train and val, compare train to val
    f1_average = f1_score(true_labels, preds, average='macro')
    #f1_per_class = f1_score(true_labels, preds, average=None)

    # class_names = encoder.inverse_transform(list(range(len(true_labels))))
    # class_names=encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))},


    if step is not None:
        if title is not None:
            wandb.log({title + ' accuracy': accuracy}, step=step)
            wandb.log({title + ' f1_average': f1_average}, step=step)
            wandb.log({title + ' confusion matrix': 
                        wandb.plot.confusion_matrix(probs=None,
                            y_true=true_labels, preds=preds,
                            class_names=encoder.classes_)}, step=step)

    # print('Accuracy: {:.3f}'.format(accuracy))
    # print('F1: {:.3f}'.format(f1))
    return accuracy, f1_average
