import logging
import sys
sys.path.append('.')

from src.utils import load_data, define_additional_args, compute_hypergeometric, configure_seeds
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
def run(data_path, num_subgraphs, num_par_samplers, use_cuda, num_iterations, eval_every, seed, sampler_args,
        training_args, model_args):

    out = logging.info

    configure_seeds(seed, 'cuda' if use_cuda else 'cpu')

    adj_full, adj_train, feats, class_arr, role = load_data(data_path, out)
    num_subgraphs_per_sampler = define_additional_args(num_subgraphs, num_par_samplers, out)
    minibatch = Minibatch(adj_full, adj_train, role, num_par_samplers, num_subgraphs_per_sampler, use_cuda,
                          sampler_args, model_args)
    trainer = Trainer(training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args)
    evaluator = Evaluator(model_args, feats, class_arr, training_args['loss'])

    if training_args['method'] == 'ours':
        total_gamma = 0
        C = trainer.C  # max sensitivity

        if training_args['distribution'] == 'hyper':
            K = (sampler_args['max_degree'] ** (model_args['num_layers'] + 1) - 1) // (sampler_args['max_degree'] - 1)
            m = sampler_args['num_root'] * (sampler_args['depth'] + 1)  # number of gradients in one batch
            gho = compute_hypergeometric(len(minibatch.node_train), K, m)

            if not sampler_args['only_roots']:
                sigma_without_C = 2 * K
                sigma_without_K = 2 * C
            else:
                gho = [gho[0], sum(gho[1:])]
                sigma_without_C = 1
                sigma_without_K = C

        elif training_args['distribution'] == 'ours':
            assert sampler_args['only_roots']
            gho_1 = sum([(sampler_args['max_degree'] + sampler_args['depth']) / (
                        len(minibatch.node_train) - i * (sampler_args['depth'] + 1)) for i in
                         range(sampler_args['num_root'] + 1)])
            gho = [1 - gho_1, gho_1]
            sigma_without_C = 1
            sigma_without_K = C

    elif training_args['method'] == 'node_dp_max_degree':
        K = (sampler_args['max_degree'] ** (model_args['num_layers'] + 1) - 1) // (
                    sampler_args['max_degree'] - 1)  # number of affected nodes in one batch
        m = sampler_args['num_nodes']  # number of nodes sampled in one batch
        C = trainer.C  # max sensitivity
        sigma_without_C = 2 * K
        sigma_without_K = 2 * C

        total_gamma = 0
        gho = compute_hypergeometric(len(minibatch.node_train), K, m)

    all_eps = []
    all_iterations = []
    all_metrics = []

    t1 = time.time()
    for it in range(1, num_iterations+1):
        if training_args['method'] == 'normal':
            trainer.train_step(*minibatch.sample_one_batch(out))
        elif training_args['method'] in ['ours', 'node_dp_max_degree']:
            trainer.dp_train_step_fast2(*minibatch.sample_one_batch(out), sigma=sigma_without_C)

            total_gamma += 1 / (training_args['alpha'] - 1) * np.log(sum(np.array([p * (
                np.exp(training_args['alpha'] * (training_args['alpha'] - 1) * (i * sigma_without_K) ** 2 / (
                            2 * (sigma_without_C * C) ** 2))) for i, p in enumerate(gho)])))

        if it % eval_every == 0:
            t2 = time.time()
            evaluator.model.load_state_dict(trainer.model.state_dict())
            preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
            metrics = evaluator.calc_metrics(preds, labels)

            all_metrics.append(metrics)
            all_iterations.append(it)

            print_statement = f"Iteration {it}:"
            for metric, val in metrics.items():
                print_statement += f"\t {metric} = {val}"
            print_statement += f"\t Training Time = {t2 - t1}"
            out(print_statement)

            if training_args['method'] in ['ours', 'node_dp_max_degree']:
                out("RDP: (" + str(training_args['alpha']) + "," + str(total_gamma) + ")")
                eps = total_gamma + np.log(1 / training_args['delta']) / (training_args['alpha'] - 1)
                out("DP: (" + str(eps) + "," + str(training_args['delta']) + ")")

                all_eps.append(eps)

            t1 = time.time()

    # Final metrics
    evaluator.model.load_state_dict(trainer.model.state_dict())
    preds, labels = evaluator.eval_step(*minibatch.sample_one_batch(out, mode='val'))
    metrics = evaluator.calc_metrics(preds, labels)

    results = metrics
    if training_args['method'] in ['ours', 'node_dp_max_degree']:
        results['gho'] = gho
        results['C'] = trainer.C.detach().cpu().numpy()
        results['gamma'] = total_gamma
        results['eps'] = total_gamma + np.log(1 / training_args['delta']) / (training_args['alpha'] - 1)
        results['all_eps'] = all_eps
        results['all_iterations'] = all_iterations
        results['all_metrics'] = all_metrics

    return results
