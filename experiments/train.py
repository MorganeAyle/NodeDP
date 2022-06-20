import logging
import sys
sys.path.append('.')

from src.utils import load_data, configure_seeds
from src.minibatch import Minibatch
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.accountant import PrivacyAccountant
from src.constants import DP_METHODS, NON_DP_METHODS, RDP_ACCOUNTANT

import time

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
def run(data_path, use_cuda, num_iterations, eval_every, seed, sampler_args,
        training_args, model_args):

    out = logging.info

    configure_seeds(seed, 'cuda' if use_cuda else 'cpu')
    adj_full, adj_train, feats, class_arr, role = load_data(data_path, out)
    minibatch = Minibatch(adj_full, adj_train, role, use_cuda, sampler_args, model_args, out)
    trainer = Trainer(training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args)
    evaluator = Evaluator(model_args, feats, class_arr, training_args['loss'], training_args['early_stopping_after'])

    if sampler_args['method'] in NON_DP_METHODS:
        accountant = None
    elif sampler_args['method'] == 'baseline':
        accountant = PrivacyAccountant(training_args, sampler_args, model_args['num_layers'], len(minibatch.node_train),
                                       trainer.C, fout=out)
    elif sampler_args['method'] in DP_METHODS:
        accountant = PrivacyAccountant(training_args, sampler_args, sampler_args['depth'], len(minibatch.node_train),
                                       trainer.C, fout=out)
    else:
        raise NotImplementedError

    t1 = time.time()
    it = 1
    while it <= num_iterations or (sampler_args['method'] in DP_METHODS and
                                   accountant.epsilon < training_args["max_eps"]):

        if sampler_args['method'] in NON_DP_METHODS:
            trainer.train_step(*minibatch.sample_one_batch(out))

        elif sampler_args['method'] in DP_METHODS:

            if accountant.epsilon > training_args["max_eps"]:
                break

            accountant.one_step()
            trainer.dp_train_step_fast(*minibatch.sample_one_batch(out), sigma=accountant.sigma)

        if it % eval_every == 0:
            t2 = time.time()
            evaluator.model.load_state_dict(trainer.model.state_dict())
            preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
            metrics = evaluator.calc_metrics(preds, labels)

            # Log results
            print_statement = f"Iteration {it}:"
            for metric, val in metrics.items():
                print_statement += f"\t {metric} = {val}"
            print_statement += f"\t Training Time = {t2 - t1}"
            out(print_statement)

            if accountant is not None:
                accountant.log(out)
            t1 = time.time()

            if evaluator.early_stopping:
                out("Early stopping...")
                break
        it += 1

    # Final metrics
    evaluator.model.load_state_dict(evaluator.best_model.state_dict())
    preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='test'))
    metrics = evaluator.calc_metrics(preds, labels)

    results = metrics
    if sampler_args['method'] in DP_METHODS:
        results['gho'] = accountant.distribution
        results['C'] = trainer.C.detach().cpu().numpy()
        if training_args['accountant'] in RDP_ACCOUNTANT:
            results['gamma'] = accountant.gamma
        results['eps'] = accountant.epsilon

    return results
