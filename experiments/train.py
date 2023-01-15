import logging
import os.path
import sys

sys.path.append('.')

from src.utils import load_data, configure_seeds
from src.minibatch import Minibatch
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.accountant import PrivacyAccountant
from src.constants import DP_METHODS, NON_DP_METHODS
from src.utils import create_save_path

import time
import math

import torch

from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(data_path, save_model_dir, use_cuda, num_iterations, eval_every, seed, sampler_args,
        training_args, model_args):
    out = logging.info

    save_model_path = create_save_path(save_model_dir, data_path, num_iterations, seed, sampler_args, training_args,
                                       model_args, out)
    configure_seeds(seed, 'cuda' if use_cuda else 'cpu')
    adj_full, adj_train, feats, class_arr, role = load_data(data_path, out)
    minibatch = Minibatch(adj_full, adj_train, role, use_cuda, sampler_args, model_args, out)
    trainer = Trainer(training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args)
    evaluator = Evaluator(model_args, feats, class_arr, training_args['loss'], training_args['early_stopping_after'])
    dataset = data_path.split('/')[-1]

    if not training_args['dp_training'] or sampler_args['method'] in NON_DP_METHODS:
        accountant = None
    elif sampler_args['method'] == 'baseline':
        accountant = PrivacyAccountant(training_args, sampler_args, model_args['num_layers'], len(minibatch.node_train),
                                       trainer.clip_norm, fout=out, num_total_nodes=adj_full.shape[0], dataset=dataset)
    elif sampler_args['method'] == 'uniform':
        accountant = PrivacyAccountant(training_args, sampler_args, None, len(minibatch.node_train),
                                       trainer.clip_norm, fout=out, num_total_nodes=adj_full.shape[0], dataset=dataset)
    elif sampler_args['method'] in DP_METHODS:
        accountant = PrivacyAccountant(training_args, sampler_args, sampler_args['depth'], len(minibatch.node_train),
                                       trainer.clip_norm, fout=out, num_total_nodes=adj_full.shape[0], dataset=dataset)
    else:
        raise NotImplementedError

    it = 1
    test_epsilons = []
    test_metrics = []
    test_metrics_best_models = []
    eps_counter = 0
    t1 = time.time()
    while it <= num_iterations or (sampler_args['method'] in DP_METHODS and
                                   accountant is not None and
                                   accountant.epsilon < training_args["max_eps"]):

        # Training
        if accountant is None:
            trainer.train_step(*minibatch.sample_one_batch(out))

        else:

            if accountant.epsilon > training_args["max_eps"]:
                break

            trainer.dp_train_step_fast(*minibatch.sample_one_batch(out), sigma=accountant.sigma)

            # update privacy budget
            accountant.one_step()
            evaluator.eps = accountant.epsilon
            if eps_counter == 0:
                eps_counter = math.ceil(evaluator.eps)

        # Evaluating
        if it % eval_every == 0:
            t2 = time.time()
            evaluator.model.load_state_dict(trainer.model.state_dict())

            preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
            metrics = evaluator.calc_metrics(preds, labels, mode='val')

            if accountant is not None:
                accountant.log(out)

            if evaluator.early_stopping:
                out("Early stopping...")
                break

            # Log results
            print_statement = f"Iteration {it}:"
            for metric, val in metrics.items():
                print_statement += f"\t {metric} = {val}"
            print_statement += f"\t Training Time = {t2 - t1}"

            t1 = time.time()
            print_statement += f"\t Evaluation Time = {t1 - t2}"
            out(print_statement)

        # Testing at certain epsilons
        if accountant is not None and evaluator.eps > eps_counter:
            evaluator.model.load_state_dict(trainer.model.state_dict())
            # save metrics on full dataset
            preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='test'))
            metrics = evaluator.calc_metrics(preds, labels, mode='test')
            test_metrics.append(metrics['F1 Micro'])
            # Update next epsilon to save
            test_epsilons.append(evaluator.eps)
            eps_counter += 1
            # Log result
            out("Test F1-Micro " + str(metrics['F1 Micro']) + " at eps=" + str(evaluator.eps))

            # Evaluate on best model if early stopping
            if evaluator.best_model:
                evaluator.model.load_state_dict(evaluator.best_model.state_dict())
                preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='test'))
                metrics = evaluator.calc_metrics(preds, labels, mode='test')
                test_metrics_best_models.append(metrics['F1 Micro'])
                # Log result
                out("Best Model Test F1-Micro " + str(metrics['F1 Micro']) + " at eps=" + str(evaluator.eps))

        it += 1

    if evaluator.best_model is None:
        evaluator.best_model = trainer.model
        if accountant is not None:
            evaluator.best_eps = accountant.epsilon

    # Final metrics
    evaluator.model.load_state_dict(evaluator.best_model.state_dict())
    preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='test'))
    metrics = evaluator.calc_metrics(preds, labels)

    # Save model
    # out("Saving model...")
    # torch.save(evaluator.best_model.state_dict(), save_model_path)

    results = metrics
    if accountant is not None:
        results['gho'] = accountant.distribution
        results['eps'] = evaluator.best_eps
        results['test_metrics'] = test_metrics
        results['test_epsilons'] = test_epsilons
        results['test_metrics_best'] = test_metrics_best_models
        results['sensitivity'] = accountant.sensitivity
        results['sigma'] = accountant.sigma

    return results
