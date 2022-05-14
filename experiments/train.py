import logging
import sys
sys.path.append('.')

from src.utils import load_data, define_additional_args, compute_hypergeometric
from src.minibatch import Minibatch
from src.trainer import Trainer
from src.evaluator import Evaluator
import numpy as np
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
def run(data_path, num_subgraphs, num_par_samplers, use_cuda, num_iterations, eval_every, sampler_args, training_args,
        model_args):

    out = logging.info

    adj_full, adj_train, feats, class_arr, role = load_data(data_path, out)
    num_subgraphs_per_sampler = define_additional_args(num_subgraphs, num_par_samplers, out)
    minibatch = Minibatch(adj_full, adj_train, role, num_par_samplers, num_subgraphs_per_sampler, use_cuda,
                          sampler_args)
    trainer = Trainer(training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args)
    evaluator = Evaluator(model_args, feats, class_arr, training_args['loss'])

    if training_args['method'] == 'ours':
        K = sampler_args['depth'] + 1  # number of affected nodes in one batch
        m = sampler_args['num_root'] * (sampler_args['depth'] + 1)  # number of nodes sampled in one batch
        C = trainer.C  # max sensitivity
        if sampler_args['only_roots']:
            sigma = 1
            gho = compute_hypergeometric(len(minibatch.node_train), K, m)
            gho = [gho[0], sum(gho[1:])]
        else:
            sigma = 2 * K
            gho = compute_hypergeometric(len(minibatch.node_train), K, m)

        total_gamma = 0

    elif training_args['method'] == 'node_dp_max_degree':
        K = sampler_args['max_degree'] + 1  # number of affected nodes in one batch
        m = sampler_args['num_nodes']  # number of nodes sampled in one batch
        C = trainer.C  # max sensitivity
        sigma = 2 * K

        total_gamma = 0
        gho = compute_hypergeometric(len(minibatch.node_train), K, m)

    t1 = time.time()
    for it in range(num_iterations):
        if training_args['method'] == 'normal':
            trainer.train_step(*minibatch.sample_one_batch(out))
        elif training_args['method'] in ['ours', 'node_dp_max_degree']:
            trainer.dp_train_step_fast(*minibatch.sample_one_batch(out), sigma=sigma)

            if not sampler_args['only_roots']:
                total_gamma += 1 / (training_args['alpha'] - 1) * np.log(sum(np.array([p * (
                    np.exp(training_args['alpha'] * (training_args['alpha'] - 1) * 2 * (i * C) ** 2 / (sigma * C) ** 2))])
                                                                             for i, p in enumerate(gho))[0])
            else:
                total_gamma += 1 / (training_args['alpha'] - 1) * np.log(sum(np.array([p * (
                    np.exp(
                        training_args['alpha'] * (training_args['alpha'] - 1) * (i * C) ** 2 / (2 * (sigma * C) ** 2)))])
                                                                             for i, p in enumerate(gho))[0])

        if it % eval_every == 0:
            t2 = time.time()
            evaluator.model.load_state_dict(trainer.model.state_dict())
            preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
            metrics = evaluator.calc_metrics(preds, labels)

            print_statement = f"Iteration {it}:"
            for metric, val in metrics.items():
                print_statement += f"\t {metric} = {val}"
            print_statement += f"\t Training Time = {t2 - t1}"
            out(print_statement)

            if training_args['method'] in ['ours', 'node_dp_max_degree']:
                out("RDP: (" + str(training_args['alpha']) + "," + str(total_gamma) + ")")
                eps = total_gamma + np.log(1 / training_args['delta']) / (training_args['alpha'] - 1)
                out("DP: (" + str(eps) + "," + str(training_args['delta']) + ")")

                if eps >= 19:
                    break

            t1 = time.time()

    # Final metrics
    evaluator.model.load_state_dict(trainer.model.state_dict())
    preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
    metrics = evaluator.calc_metrics(preds, labels)

    results = metrics
    results['iterations'] = it
    if training_args['method'] in ['ours', 'node_dp_max_degree']:
        results['gho'] = gho
        results['C'] = trainer.C.detach().cpu().numpy()
        results['alpha'] = training_args['alpha']
        results['gamma'] = total_gamma
        results['eps'] = total_gamma + np.log(1 / training_args['delta']) / (training_args['alpha'] - 1)
        results['delta'] = training_args['delta']

    return results
