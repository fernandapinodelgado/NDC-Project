import torch
from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup


def load_model(bert_version):
    """Loads a pretrained BERT model and sends it to GPU/TPU if possible.

    Args:
        bert_version (str): The version of BERT to be used.

    Returns:
        Tuple[BertForSequenceClassification, device]: The pretrained BERT model
            and the device it was sent to (if any).
    """
    config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=bert_version, num_labels=12)
    
    print('Config type:', str(type(config)), '\n')
    print(config)

    model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=bert_version, config=config)

    print('\nModel type:', str(type(model)))

    if torch.cuda.is_available():
        print('\nLoading model to GPU...')
        device = torch.device('cuda')
        print('   GPU:', torch.cuda.get_device_name(0))
        desc = model.to(device)
    else:
        print('\nERROR!!! CUDA not available.')
        exit(1)
    print('   DONE.')

    return (model, device)


def get_optimizer_scheduler(model, py_inputs, wandb):
    """Creates and returns an AdamW optimizer and a linear scheduler with
    warmup, from a given model.

    Args:
        model (BertForSequenceClassification): The BERT model to be used
        num_batches (int): The number of batches
        wandb (TODO): The WandB object being used for logging

    Returns:
        Tuple[AdamW, TODO]: The generated optimizer and scheduler
    """
    optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate,
                      eps=1e-8)

    epochs = wandb.config.epochs

    total_steps = len(py_inputs) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=wandb.config.warmup_steps,
                    num_training_steps=total_steps)

    return (optimizer, scheduler)


def get_learning_rate(optimizer):
    """Returns the learning rate of a given optimizer."""
    # TODO: How does this work?
    for param_group in optimizer.param_groups:
        return param_group['lr']
