import wandb
import numpy as np
from .data_loader import MNISTLoader
from ann import NeuralNetwork
from ann import Optimizer

sweep_config = {
    'method' : 'random',
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'learning_rate': {'distribution': 'log_uniform_values',
                          'min': 1e-4, 'max': 1e-1},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 
                                 'adam','nadam']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
        'weight_init': {'values': ['random', 'xavier']},
        'loss': {'values': ['mean_squared_error', 'cross_entropy']},
        'hidden_size': {'values': [
            [16, 16, 16],
            [32, 32, 32],
            [64, 64, 64],
            [128, 128, 128],
            [64, 64, 32],
            [128, 64, 32],
            [128, 128, 64, 64],
        ]},
        'epochs': {'value': 10},
    }
}

def train_with_wandb_sweep():

    dls = MNISTLoader('mnist',val_split=0.2)
    with wandb.init() as run:
        
        config = run.config

        dls.batch_size = config.batch_size
        model = NeuralNetwork(config.hidden_size, config.weight_init,
                                config.activation, config.loss)
        optimizer = Optimizer(config.optimizer,model.layers)
        logs = model.train(dls,optimizer,
                                    config.epochs,config.learning_rate)

        run.define_metric("epoch/*", step_metric = "epoch/num")
        run.define_metric("batch/*", step_metric = "batch/step")

        for i, loss in enumerate(logs['raw_loss']):
            if i%10 == 0:
                wandb.log({"batch/raw_loss": loss, "batch/step": i})

        for e in range(len(logs['train_loss'])):
            wandb.log({
                "epoch/train_loss": logs['train_loss'][e],
                "epoch/valid_loss": logs['valid_loss'][e],
                "epoch/accuracy": logs['accuracy'][e],
                "epoch/f1_score": logs['f1_macro'][e],
                "epoch/time": logs['time'][e],
                "epoch/num": e+1,
            })

        run.summary["best_val_accuracy"] = max(logs['accuracy'])
        run.summary["best_f1_score"] = max(logs["f1_macro"])

        arch_label = "-".join(map(str, config.hidden_size))
        run.summary["arch-string"] = arch_label,
        run.summary["arch-depth"] = len(config.hidden_size)
        run.summary["num_neurons"] = sum(config.hidden_size)

    return