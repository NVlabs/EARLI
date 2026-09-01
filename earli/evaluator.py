# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import os
import pickle
import random
import time
import warnings
from collections import defaultdict

import hygese
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import wandb

from .buffer import RolloutBuffer
from .external_solvers import HGS
from .utils.evaluation_utils import update_stats, update_batch_log
from .models.attention_model import PosAttentionModel
from .models.sampler import Sampler
from .self_play import optimize_torch, SelfPlay
from .utils import analysis_utils, evaluation_utils
from .utils.safe_serialization import safe_torch_load
from .utils.nv import seed_all, printable_time, find_largest_common_prefixes


class Evaluator(object):
    """

    """

    def __init__(self, env_class, config, logger, model=None, fabric=None):
        # Load the game and the config from the module with the game name
        self.best_model_path = None
        optimize_torch()
        self.config = config
        self.use_fabric = fabric is not None
        self.fabric = fabric
        model_class = PosAttentionModel
        observation_space, action_space = env_class(config, datafile=config['eval']['data_file']).spaces
        self.model = model
        self.train_phase = 0
        self.use_ray = config['speedups']['use_ray']
        self.n_workers = self.num_workers()
        self.logger = logger
        self.worker_id = self.fabric.global_rank if self.use_fabric else 0
        seed_all(config['train']['seed'])
        _, n_problems = self.get_n_iterations_and_n_problem_per_env(total_problems=self.config['muzero']['data_steps_per_epoch'],
                                                                    stage_label='train', strict=False)
        self.dummy_env = {'test': env_class(config, datafile=config['eval']['data_file'], env_type='eval')}
        self.n_problems = {k: env.dataset_size for k, env in self.dummy_env.items()}
        self.n_problems['train'] = self.config['muzero']['data_steps_per_epoch']

        sampler = Sampler(self.config)
        if self.model is None:
            self.model = model_class(observation_space, action_space, config=config, sampler=sampler)

        def test_constructor(stage, **kwargs):
            assert stage == 'test'
            data_file = config['eval']['data_file']
            _, n_problems = self.get_n_iterations_and_n_problem_per_env(
                    total_problems=self.n_problems[stage],
                    stage_label=stage, strict=True)
            return SelfPlay(Game=env_class, config=self.config,
                            seed=config['train']['seed'], datafile=data_file,
                            model=self.model, sampler=sampler, n_problems=n_problems,
                            n_beams=config['train']['n_beams'], env_type='eval', worker_id=self.worker_id,
                            n_workers=self.n_workers, **kwargs)

        # test env
        self.tester = test_constructor(stage='test')

        if self.config['eval']['max_problems'] is not None:
            for k, v in self.config['eval']['max_problems'].items():
                if v < self.n_problems[k]:
                    self.n_problems[k] = v
        self.replay_buffer = RolloutBuffer(config=self.config)

        # Verify all dataset-sizes are divisible by n_parallel_problems
        for key, n_problem in self.n_problems.items():
            if key == 'train':
                continue
            n_parallel = min(self.config['train']['n_parallel_problems'], n_problem)
            assert n_problem % n_parallel == 0, \
                f'The number of {key} problems ({n_problem}) must be divisible by n_parallel_problems ({n_parallel}).'
            n_data_collection_iterations = n_problem // n_parallel
            if self.use_ray:
                assert n_data_collection_iterations % self.config['speedups']['n_workers'] == 0, \
                    f'The number of {key} batches ({n_data_collection_iterations}) must be divisible by n_workers ({n_parallel}).'

        self.gae_lambda = 1

        # Initialize test logs
        self.test_stats = defaultdict(lambda: dict())
        self.test_log = defaultdict(lambda: list())
        self.test_stats['problems_path'] = self.config['eval']['data_file']

        self.auto_resume_history = []

    def inference(self):
        """

        """
        self.load_model(self.config['train']['pretrained_fname'], restore_state=False)
        self.collect_data(0, 0, stage='test',
                          detailed_log=2 if self.config['eval']['detailed_test_log'] else 0)
        self.logger.dump(step=1, fabric=self.fabric, force=True)
        self.save_logs()
        return [0, self.model]




    def record(self, stage, key, value):
        self.logger.record(stage, key, value)

    def log_results(self, stage='test', detailed=False, stats=None):
        if stats is None:
            stats = self.test_stats
        if stats['game_iters']:
            self.logger.record(stage, "mean_game_iters", np.mean(list(stats['game_iters'].values())))
        self.logger.record(stage, "mean_game_clocktime", stats['game_clocktime'])

        self.logger.record(stage, "total_games", stats['total_games'])
        if stats['number_of_vehicles']:
            self.logger.record(stage, "number_of_vehicles", np.mean(list(stats['number_of_vehicles'].values())))

        all_recent_returns = list(stats['recent_returns'].values())
        recent_return = np.mean(all_recent_returns)
        self.logger.record(stage, "mean_reward", recent_return)
        self.logger.record(stage, "std_reward", np.std(all_recent_returns))

        if 'all_routes' in stats:
            if self.config['eval']['apply_local_search']:
                stats['all_ls_routes'] = self.apply_local_search_multi(
                        list(stats['all_routes'].values()), self.dummy_env[stage])

        if 'baseline_returns' in stats and not np.isclose(list(stats['baseline_returns'].values())[0], 0) and \
                not np.isnan(list(stats['baseline_returns'].values())[0]):
            best_key = 'best_returns'
            best_return = np.mean(list(stats[best_key].values()))
            net_solver_returns = np.array(list(stats['baseline_returns'].values()))
            solver_returns = net_solver_returns
            if 'baseline_vehicles' in stats:
                car_penalty = self.dummy_env[stage].extra_car_penalty
                solver_returns = solver_returns - car_penalty * np.array(list(stats['baseline_vehicles'].values()))
            solver_return = np.mean(solver_returns)
            all_gaps = [r / s for r, s in zip(all_recent_returns, solver_returns)]

            # for the *current* model: mean return / baseline mean return
            self.logger.record(stage, "gap_to_optimality", float(recent_return / solver_return))
            # for the *best* model: mean return / baseline mean return
            self.logger.record(stage, "best_iter_gap_to_optimality", float(best_return / solver_return))
            # for the *current* model, for the *best* env: return / baseline return
            self.logger.record(stage, "best_env_gap_to_optimality", float(min(all_gaps)))

            env = self.dummy_env[stage]
            reward_normalization = env.reward_normalization
            distances = env.data['distance_matrix']

            if 'all_routes' in stats and len(stats['all_routes']) > 0:
                # analyze pre-ls solutions
                self.solution_set_analysis(
                        list(stats['all_routes'].values()), stage, distances, net_solver_returns / reward_normalization,
                        prefix='top_k', det_prefix='deterministic')

                if self.config['eval']['apply_local_search']:
                    self.solution_set_analysis(
                            stats['all_ls_routes'], stage, distances, net_solver_returns / reward_normalization,
                            prefix='ls_k', det_prefix='ls_deterministic')

            net_returns = None
            if 'baseline_vehicles' in stats:
                net_returns = list(stats['net_returns'].values())
                equal_vehicles = [nr == ns for nr, ns in zip(
                        stats['number_of_vehicles'].values(), stats['baseline_vehicles'].values())]
                conditioned_gap = np.mean([r for r, eq in zip(net_returns, equal_vehicles) if eq]) / np.mean(
                        [s for s, eq in zip(net_solver_returns, equal_vehicles) if eq]) \
                    if np.any(equal_vehicles) else np.nan
                conditioned_neq_gap = np.mean([r for r, eq in zip(net_returns, equal_vehicles) if not eq]) / np.mean(
                        [s for s, eq in zip(net_solver_returns, equal_vehicles) if not eq]) \
                    if not np.all(equal_vehicles) else np.nan
                # for the *current* model: vehicles / baseline vehicles
                rlopt_vehicles = np.array(list(stats['number_of_vehicles'].values()))
                baseline_vehicles = np.array(list(stats['baseline_vehicles'].values()))
                self.logger.record(stage, "vehicle_mean_gap", float(
                        np.mean(rlopt_vehicles) / np.mean(baseline_vehicles)))
                self.logger.record(stage, "vehicle_optimal_ratio", float(np.mean(rlopt_vehicles <= baseline_vehicles)))
                # for the *current* model, for all envs with equal vehicles: mean return / baseline mean return
                self.logger.record(stage, "conditioned_eq_cost_gap", float(np.mean(conditioned_gap)))
                self.logger.record(stage, "conditioned_neq_cost_gap", float(np.mean(conditioned_neq_gap)))

            if detailed:
                if not stage.startswith('test'):
                    warnings.warn(f'Detailed logging may be faulty for stage {stage}!=test')

                # apply local search to best solutions
                if self.config['eval']['apply_local_search']:
                    ls_vehicles, ls_costs = self.apply_local_search(
                        solutions=list(stats['optimal_route'].values()), env=env)
                    equal_vehicles = [ls == base for ls, base in zip(
                        ls_vehicles, stats['baseline_vehicles'].values())]
                    ls_gaps = np.array(ls_costs) / (-net_solver_returns / reward_normalization)
                    conditioned_ls_gaps = [gap for gap, eq in zip(ls_gaps, equal_vehicles) if eq] \
                        if np.any(equal_vehicles) else [np.nan]
                    self.logger.record(stage, "post_ls_cond_cost_gap", np.mean(conditioned_ls_gaps))
                    self.logger.record(stage, "post_ls_optimal_vehicles", float(np.mean(ls_vehicles <= baseline_vehicles)))

                # plot optimality-gap distribution over problems
                plt.figure()
                all_returns = dict(rlopt=net_returns)
                all_vehicles = dict(rlopt=list(stats['number_of_vehicles'].values()))
                if 'naive_solutions' in env.data:
                    for b, solutions in env.data['naive_solutions'].items():
                        if b != 'random':
                            all_returns[b] = - reward_normalization * np.array(solutions['costs'])
                            all_vehicles[b] = [np.sum(np.array(sol) == 0) - 1 for sol in solutions['solutions']]
                ref_vehicles = list(stats['baseline_vehicles'].values())
                analysis_utils.analyze_test_solutions(all_returns, all_vehicles, net_solver_returns, ref_vehicles)
                if hasattr(self.logger, 'wandb_logger'):
                    self.logger.wandb_logger.log({f'[{stage}] Optimality gap distribution': wandb.Image(plt)}, commit=False)

                # show sample solutions
                ids = np.argmin(all_gaps), np.random.randint(self.n_problems[stage]), np.argmax(all_gaps)
                plt.figure()

                axs = analysis_utils.Axes(2 * len(ids), 2, (8, 6.5))
                for i, i_problem in enumerate(ids):
                    pos = env.data['positions'][i_problem]
                    dist = env.data['distance_matrix'][i_problem]

                    sol = env.data['baseline_solutions_lists'][i_problem]
                    n_vehicles = np.sum(np.array(sol) == 0) - int(self.config['problem_setup']['last_return_to_depot'])
                    cost = evaluation_utils.solution_cost(sol, dist, self.config['problem_setup']['distance_Lp_norm'])
                    xx, yy = pos[sol, 0], pos[sol, 1]
                    analysis_utils.show_trajectory(xx, yy, sol, ax=axs[2 * i + 0], colorbar=False)
                    axs[2 * i + 0].set_aspect('equal')
                    axs.labs(2 * i + 0, '', '', f'[cuopt] problem #{i_problem:d}: vehicles={n_vehicles:.0f}, distance={cost:.3f}',
                             fontsize=15)

                    sol = list(stats['optimal_route'].values())[i_problem].tolist()
                    n_vehicles = np.sum(np.array(sol) == 0) - int(self.config['problem_setup']['last_return_to_depot'])
                    cost = evaluation_utils.solution_cost(sol, dist, self.config['problem_setup']['distance_Lp_norm'])
                    xx, yy = pos[sol, 0], pos[sol, 1]
                    analysis_utils.show_trajectory(xx, yy, sol, ax=axs[2 * i + 1], colorbar=False)
                    axs[2 * i + 1].set_aspect('equal')
                    axs.labs(2 * i + 1, '', '', f'[rlopt] problem #{i_problem:d}: vehicles={n_vehicles:.0f}, distance={cost:.3f}',
                             fontsize=15)
                plt.tight_layout()
                if hasattr(self.logger, 'wandb_logger'):
                    self.logger.wandb_logger.log({f'[{stage}] Sample solution': wandb.Image(plt)}, commit=False)

        return

    def apply_local_search(self, solutions, env, verify_valid_solution=True, verbose=1):
        n_problems = len(solutions)
        ls_costs, ls_vehicles = [], []
        for ind in range(n_problems):
            # original solution
            sol = solutions[ind].tolist()
            vehc1 = np.sum(np.array(sol[:-1]) == 0)

            # env
            scale = 1e3 / env.radius
            distance_matrix = (scale * env.data['distance_matrix'][ind]).tolist()
            demands = env.data['demand'][ind].tolist()
            capacity = env.data['capacity'][ind].item()

            # apply ls
            ls_sol, vehc2, cost2 = HGS.local_search([sol], distance_matrix, demands, capacity)
            ls_sol, vehc2, cost2 = ls_sol[0], vehc2[0], cost2[0]

            if verify_valid_solution:
                if not evaluation_utils.verify_solution(ls_sol, np.array(demands), capacity, max_vehicles=None):
                    if verbose >= 1:
                        warnings.warn(f'Local-search returned infeasible solution for problem {ind}.')
                    ls_sol, vehc2 = sol, vehc1

            if vehc1 != vehc2:
                warnings.warn(f'local search on problem {ind} modified vehicles from {vehc1} to {vehc2}.')

            ls_vehicles.append(vehc2)
            ls_costs.append(evaluation_utils.solution_cost(ls_sol, env.data['distance_matrix'][ind]))

        return np.array(ls_vehicles), np.array(ls_costs)

    def apply_local_search_multi(self, solutions, env, verify_valid_solutions=True, verbose=1):
        n_problems = len(solutions)
        ls_sols, ls_times = [], []
        solver = hygese.Solver()
        for ind in range(n_problems):
            # original solutions
            sols = solutions[ind]

            # env
            scale = 1e3 / env.radius
            distance_matrix = (scale * env.data['distance_matrix'][ind]).tolist()
            demands = env.data['demand'][ind].tolist()
            capacity = env.data['capacity'][ind].item()

            # apply ls
            t0 = time.time()
            ls_sol, vehc2, cost2 = HGS.local_search(sols, distance_matrix, demands, capacity,
                                                    solver=solver)
            ls_times.append(time.time() - t0)

            if verify_valid_solutions:
                ls_sol = [sol for sol in ls_sol if evaluation_utils.verify_solution(
                    sol, np.array(demands), capacity, max_vehicles=None, verbose=verbose>=2)]
                if len(ls_sol) == 0 and verbose >= 1:
                    warnings.warn(f'Local-search returned only infeasible solutions for problem {ind}.')

            ls_sols.append(ls_sol)

        if verbose >= 1:
            print(f'\nLocal-search duration per problem (for {len(sols)} solutions): '
                  f'{np.mean(ls_times)} +- {np.std(ls_times)/np.sqrt(n_problems)} [s]')

        return ls_sols

    def solution_set_analysis(self, solutions, stage, distances, normalized_net_returns,
                              prefix='top_k', det_prefix='deterministic'):
        n_solutions = len(solutions[0])
        if n_solutions < 2:
            return

        deterministic_solution_costs = [
            (evaluation_utils.solution_cost(problem_sols[0], dist, self.config['problem_setup']['distance_Lp_norm']) \
            if len(problem_sols)>0 else np.nan)
            for problem_sols, dist in zip(solutions, distances)
        ]
        deterministic_solution_gaps = np.array(deterministic_solution_costs) / (-normalized_net_returns)
        self.logger.record(stage, f"{det_prefix}_solution_cost_gap", np.mean(deterministic_solution_gaps))

        top_k_mean_costs = [
            np.mean([evaluation_utils.solution_cost(sol, dist, self.config['problem_setup']['distance_Lp_norm'])
                     for sol in problem_sols])
            for problem_sols, dist in zip(solutions, distances)]
        top_k_mean_gaps = np.array(top_k_mean_costs) / (-normalized_net_returns)
        self.logger.record(stage, f"{prefix}_mean_cost_gap", np.mean(top_k_mean_gaps))

        edge_cover, edge_sharing = analysis_utils.edge_coverage_analysis(solutions)
        self.logger.record(stage, f"{prefix}_edge_coverage", edge_cover)
        self.logger.record(stage, f"{prefix}_mean_edge_sharing", edge_sharing)


    def get_args_to_collect_data(self, stage='test'):
        assert stage.startswith('test')
        deterministic = self.config['eval']['deterministic_test_beam']
        n_data_collection_iterations, n_problems_per_env = (
            self.get_n_iterations_and_n_problem_per_env(total_problems=self.n_problems[stage], stage_label=stage, log=False))
        if stage == 'test':
            log, stats = self.test_log, self.test_stats
            data_collector = self.tester
        else:
            raise ValueError(stage)

        return data_collector, log, stats, n_data_collection_iterations, deterministic

    def get_n_iterations_and_n_problem_per_env(self, total_problems, stage_label, strict=True, log=True):
        assert not (self.use_ray and self.use_fabric)
        if strict:
            assert total_problems % self.n_workers == 0, (
                f"The number of problems in stage {stage_label} is not divisible by the number "
                f"of workers {self.n_workers}. Adjust the number of workers accordingly.")
        n_problems_per_env = min(total_problems // self.n_workers, self.config['train']['n_parallel_problems'])
        n_data_collection_iterations = total_problems // (n_problems_per_env * self.n_workers)
        if log:
            logging.info(f"Number of workers in {stage_label} is set to {self.n_workers}, "
                         f"with {n_data_collection_iterations} data collection iterations, and {n_problems_per_env} parallel problem per env")
        return n_data_collection_iterations, n_problems_per_env

    def num_workers(self):
        if self.use_ray:
            n_workers = self.config['speedups']['n_workers']
        elif self.use_fabric:
            n_workers = self.fabric.world_size
        else:
            n_workers = 1
        return n_workers

    def collect_data(self, training_step, train_env_steps, stage='test', detailed_log=0,
                     data_collector=None, log=None, stats=None, n_data_collection_iterations=None,
                     deterministic=None):
        if data_collector is None:
            data_collector, log, stats, n_data_collection_iterations, deterministic = \
                self.get_args_to_collect_data(stage)
        env_steps = 0
        restart_iteration_log = True
        t_data = time.time()
        returns = []
        cars = []
        ids = []
        for i in range(n_data_collection_iterations):
            t0 = time.time()
            game_histories, infos = data_collector.play_game(deterministic=deterministic, detailed_log=detailed_log)
            total_time = time.time() - t0
            n_games = len(infos['env_id'])
            infos['game_clocktime'] = total_time / n_games
            returns.extend(infos['best_return'].tolist())
            if 'num_vehicles' in infos and infos['num_vehicles'] is not None:
                cars.extend(infos['num_vehicles'].tolist())
                ids.extend(infos['env_id'].tolist())
            env_steps += infos['env_steps'].sum().item()
            stats = update_stats(train_env_steps, infos, stats, restart_iteration_log,
                                 n_workers=self.n_workers, worker_id=self.worker_id)
            restart_iteration_log = False

            if stage == 'test':
                print(f'{stage} time part {i + 1}/{n_data_collection_iterations}: {printable_time(t_data)}')
        update_batch_log(stats, self.replay_buffer.get_usage())
        self.log_results(stage=stage, detailed=detailed_log >= 2, stats=stats)
        return np.mean(returns), np.mean(cars) if len(cars) > 0 else None, env_steps

    def merge_stats(self, stats):
        all_stats = defaultdict(lambda: dict())
        for key in ['total_games', 'data_samples', 'total_env_steps']:
            all_stats[key] = self.fabric.all_reduce(torch.tensor(stats[key]), reduce_op='sum')
        for key in ['game_clocktime']:
            all_stats[key] = self.fabric.all_reduce(torch.tensor(stats[key]), reduce_op='max')
        for key in ('baseline_returns', 'baseline_vehicles', 'recent_returns', 'best_returns', 'game_iters', 'net_returns',
                    'number_of_vehicles', 'optimal_route', 'all_routes'):  # merging dictionaries
            res = [None] * self.n_workers
            torch.distributed.all_gather_object(res, stats[key])
            all_stats[key] = {}
            for d in res:
                all_stats[key].update(d)
        return all_stats


    def get_logs(self):
        try:
            test_log = pd.DataFrame(self.test_log)
        except:
            print({k: len(v) for k, v in self.test_log})
            raise
        return test_log, self.test_stats


    def save_logs(self, dir_path='./outputs', remove_stats_keys=True):
        test_info, test_stats = self.get_logs()
        if 'tree_nodes' in test_stats and test_stats['tree_nodes']:
            test_stats = test_stats.copy()
            test_stats['tree_nodes'] = 'DELETED'
        if remove_stats_keys:
            test_stats = self.get_printable_stats(test_stats)
        else:
            test_stats = {str(k): v for k, v in test_stats.items()}
        if 'optimal_route' in test_stats:
            test_stats['optimal_route'] = [sol.tolist() for sol in test_stats['optimal_route']]
        if 'all_routes' in test_stats:
            test_stats['all_routes'] = [[sol.tolist() for sol in prob_sols] for prob_sols in test_stats['all_routes']]
        test_stats['reward_normalization'] = self.dummy_env['test'].reward_normalization if hasattr(
            self.dummy_env['test'], 'reward_normalization') else None
        test_stats['returns'] = test_stats['recent_returns']
        del test_stats['recent_returns'], test_stats['best_returns']

        if dir_path is None:
            dir_path = wandb.run.dir
        fpath = os.path.join(dir_path, 'test_logs.pkl')
        with open(fpath, 'wb') as fd:
            print(f'Saving logs to: {fpath}')
            pickle.dump(test_stats, fd)

    def get_printable_stats(self, stats):
        stats2 = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                stats2[k] = list(v.values())
            else:
                stats2[k] = v
        return stats2

    def load_model(self, filename='model.m', restore_state=True):
        logging.info(f'Loading model weights from {filename}')
        if self.use_fabric:
            checkpoint = self.fabric.load(filename)
        else:
            checkpoint = safe_torch_load(filename)
        iter_data = None
        if 'model_state_dict' in checkpoint:  # new file format
            model_params = checkpoint['model_state_dict']
            if restore_state:
                self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_worker.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                np.random.set_state(checkpoint['numpy_rng_state'])
                torch.set_rng_state(checkpoint['torch_rng_state'])
                random.setstate(checkpoint['python_rng_state'])
                iter_data = checkpoint['iter_data']
                self.test_stats.update(checkpoint['stats']['test_stats'])
                self.best_model_path = checkpoint.get('best_model_path', None)
                self.auto_resume_history = checkpoint.get('auto_resume_history', [])
                self.training_worker.scaler.load_state_dict(checkpoint['scaler'])
        else:
            model_params = checkpoint
        model_params = {k.replace('._orig_mod', ''): v for k, v in model_params.items()}  # adding uncompiled keys
        # Check the keys
        for model_key in self.model.state_dict().keys():
            cleared_key = model_key.replace('._orig_mod', '')
            if cleared_key in model_params:
                model_params[model_key] = model_params.pop(cleared_key)
        missing_keys, unexpected_keys = self.model.load_state_dict(model_params, strict=False)
        if missing_keys:
            missing_keys_modules = find_largest_common_prefixes(missing_keys)
            logging.warning(f'Missing keys in the pretrained model: {missing_keys_modules}')
        if unexpected_keys:
            unexpected_keys_modules = find_largest_common_prefixes(unexpected_keys)
            logging.warning(f'Unexpected keys in the pretrained model (missing in the trained model): {unexpected_keys_modules}')
        return iter_data
