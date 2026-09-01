# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import collections
import os
import pickle as pkl
import tempfile
import time
import warnings

import cudf
import numpy as np
import pandas as pd

import yaml
from .unified_logger import UnifiedLogger

import cuopt
from .cuopt_solver.cuopt_solver import CuOptSolver

import wandb
from .utils import analysis_utils as utils
from .utils import evaluation_utils as eval_utils
from .utils.nv import verify_consistent_config
from .utils.safe_serialization import safe_pickle_load


def load_problems(problems_path, problems_range, config):
    with open(problems_path, 'rb') as hh:
        problems = safe_pickle_load(hh)
    problem_size = problems['distance_matrix'].shape[-1]
    radius = np.max(np.abs(problems['positions']))
    if config['cuopt']['normalization'] is None:
        config['cuopt']['normalization'] = np.sqrt(radius)
    CUOPT_NORMALIZATION = config['cuopt']['normalization']
    LP = 2
    if not 'demand' in problems:
        if 'demands' in problems:
            problems['demand'] = problems['demands']
        else:
            raise ValueError('Demand not found in problems')
    if not 'capacity' in problems:
        if 'capacities' in problems:
            problems['capacity'] = problems['capacities']
        else:
            raise ValueError('Capacity not found in problems')
    max_problems = problems['demand'].shape[0]
    if problems_range is None:
        problems_range = (0, max_problems)
    else:
        if problems_range[1] > max_problems:
            problems_range[1] = max_problems
        if problems_range[0] >= problems_range[1]:
            raise ValueError(problems_range)
    n_problems = problems_range[1] - problems_range[0]
    print(f'Problems: {problems_range[0]}-{problems_range[1]-1}')
    return problems, problems_range, problem_size, n_problems, CUOPT_NORMALIZATION, LP


def download_wandb_file(run, user, project, filename='test_logs.pkl'):
    api = wandb.Api()
    run = api.run(f"{user}/{project}/{run}")
    file = run.file(filename)

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    file.download(root=temp_dir, replace=True)

    return file_path


def get_initial_solutions(config, problems, problems_range, METHODS, SOLUTION_PATHS, NAIVE_SOURCE,
                          MAX_EXTERNAL_SOLUTIONS, CHOOSE_EXTERNAL_SOLUTIONS, delete_solutions, LP):
    rlopt_solutions, rlopt_vehicles, rlopt_costs = {}, {}, {}
    rl_methods = [method for method in METHODS if 'RL' in method]
    # if len(rl_methods) > 1:
    #     raise NotImplementedError()
    # if len(rl_methods) > 1 and len(SOLUTION_PATHS) == 1:
    #     SOLUTION_PATHS = len(rl_methods) * SOLUTION_PATHS
    # if len(rl_methods) > 0:
    #     for method, sol_path in zip(rl_methods, SOLUTION_PATHS):
    #         rlopt_solutions[method], rlopt_vehicles[method], rlopt_costs[method] = load_external_solutions(
    #             sol_path, problems['distance_matrix'], MAX_EXTERNAL_SOLUTIONS, to_sort=True,
    #             post_ls_solutions=config['injection']['post_ls_solutions']>=1)
    if len(rl_methods) > 0:
        sols, vehcs, costs = load_external_solutions(
            SOLUTION_PATHS, problems, MAX_EXTERNAL_SOLUTIONS, to_sort=True,
            post_ls_solutions=config['injection']['post_ls_solutions']>=1, get_to_choose=CHOOSE_EXTERNAL_SOLUTIONS, LP=LP)
        for method in rl_methods:
            rlopt_solutions[method], rlopt_vehicles[method], rlopt_costs[method] = sols, vehcs, costs
        if delete_solutions:
            for sol_path in set(SOLUTION_PATHS):
                os.unlink(sol_path)
    else:
        rlopt_solutions, rlopt_vehicles, rlopt_costs = {}, {}, {}

    # generate naive solutions
    naive_solutions = []
    if np.any([('naive' in method or 'RL' in method) for method in METHODS]):
        if NAIVE_SOURCE is not None:
            # try loading
            NAIVE_PATH = download_wandb_file(NAIVE_SOURCE, filename='test_logs.pkl')
            naive_solutions, _, _ = load_external_solutions(
                [NAIVE_PATH], problems, 1, to_sort=True,
                post_ls_solutions=config['injection']['post_ls_solutions']>=2, LP=LP)
            naive_solutions = naive_solutions[problems_range[0]:problems_range[1]]
            os.unlink(NAIVE_PATH)
        # else:
        #     # generate
        #     naive_solutions = get_naive_solutions(problems, N_NAIVE_SOLUTIONS, problems_range,
        #                                           use_greedy=config['injection']['use_greedy'])

    return rlopt_solutions, rlopt_vehicles, rlopt_costs, naive_solutions


def load_solutions_from_path(fpath, post_ls_solutions=True):
    with open(fpath, 'rb') as hh:
        pp = safe_pickle_load(hh)

    # extract solutions
    keys_priority = ['all_routes', 'top_k_paths']
    if post_ls_solutions:
        keys_priority = ['all_ls_routes'] + keys_priority
    for k in keys_priority:
        if k in pp:
            key = k
            break
        else:
            warnings.warn(f'"{k}" not available in logs.')
    else:
        raise IOError(f'Solutions are not available in logs.')

    solutions = pp[key]
    return solutions

def load_external_solutions(logs_fpaths, problems, K=None, to_sort=True, force_opt_cars=False, opt_cars=None,
                            post_ls_solutions=True, get_to_choose=False, LP=2):
    # load results
    solutions = [load_solutions_from_path(fpath, post_ls_solutions) for fpath in logs_fpaths]  # solutions[source][problem][sol][node]
    solutions = [list(src_sols) for src_sols in zip(*solutions)]  # solutions[problem][source][sol][node]

    # take top k
    if not get_to_choose:
        if isinstance(K, (list, tuple)):
            solutions = [[src_sols[:K[i_src]] for i_src, src_sols in enumerate(prob_sols)]
                         for prob_sols in solutions]
            solutions = [[sol for src_sols in prob_sols for sol in src_sols] for prob_sols in solutions]  # solutions[problem][sol][node]
        else:
            solutions = [[sol for src_sols in prob_sols for sol in src_sols] for prob_sols in solutions]  # solutions[problem][sol][node]
            if K is not None:
                solutions = [prob_sols[:K] for prob_sols in solutions]
    else:
        solutions = [[sol for src_sols in prob_sols for sol in src_sols] for prob_sols in solutions]  # solutions[problem][sol][node]

    # remove invalid solutions
    solutions = [[sol for sol in problem_solutions if eval_utils.verify_solution(sol, prob_demands)]
                 for problem_solutions, prob_demands, prob_cap in zip(solutions, problems['demand'], problems['capacity'])]

    # sort solutions
    if to_sort or force_opt_cars:
        cars = [[np.sum([node==0 for node in sol[:-1]]) for sol in problem_solutions]
                for ind, problem_solutions in enumerate(solutions)]
        costs = [[eval_utils.solution_cost(sol, problems['distance_matrix'][ind], LP) for sol in problem_solutions] for ind, problem_solutions in enumerate(solutions)]
        if force_opt_cars:
            if opt_cars is None:
                opt_cars = [prob_cars[0] for prob_cars in cars]
        else:
            opt_cars = len(cars) * [np.inf]
        solutions = [[x for c, _, x in sorted(zip(problem_cars, problem_costs, problem_solutions)) if c <= problem_opt]
                     for problem_cars, problem_costs, problem_solutions, problem_opt in zip(cars, costs, solutions, opt_cars)]

    if get_to_choose:
        if isinstance(K, (list, tuple)):
            raise ValueError
        if K is not None:
            solutions = [prob_sols[:K] for prob_sols in solutions]

    # save best costs
    cars = [np.sum([node==0 for node in prob_sols[0][:-1]]) if prob_sols else np.inf for ind, prob_sols in enumerate(solutions)]
    costs = [eval_utils.solution_cost(prob_sols[0], problems['distance_matrix'][ind], LP) if prob_sols else np.inf for ind, prob_sols in enumerate(solutions)]

    return solutions, cars, costs


def preprocess_external_solutions(method, ind, problems, n_carriers, rlopt_solutions, naive_solutions, problems_range, avoid_suboptimal_cars=None, force_feasible=False, LP=2,
                                  BOTH_RL_AND_NAIVE=0, NAIVE_RUNTIME=0, USE_REF_VEHICLES=False, ENV='vrp', SORT_SOLUTIONS='cost', SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT=False):
    external_solutions = []
    extra_time_needed = 0
    if 'RL' in method:
        external_solutions = rlopt_solutions[method][ind]
        if force_feasible:
            external_solutions = [sol for sol in external_solutions if eval_utils.verify_solution(
                sol, problems['demand'][ind], problems['capacity'][ind].item(), n_carriers)]
        add_naive = False
        if BOTH_RL_AND_NAIVE == 2:  # add naive anyway
            add_naive = True
        elif BOTH_RL_AND_NAIVE == 1:  # add naive if rl is suboptimal
            if len(external_solutions) == 0:
                add_naive = True
            else:
                best_cars = np.min([np.sum(np.array(sol[:-1]) == 0) for sol in external_solutions])
                is_optimal_cars = best_cars == (n_carriers if USE_REF_VEHICLES else (
                        np.sum(problems['demand'][ind]) / int(np.ceil(problems['capacity'][ind]))))
                add_naive = not is_optimal_cars
        if add_naive:
            extra_time_needed = NAIVE_RUNTIME
            external_solutions = external_solutions + naive_solutions[ind - problems_range[0]]
    elif 'naive' in method:
        external_solutions = naive_solutions[ind - problems_range[0]]
    else:
        return external_solutions, np.nan, np.nan, 0

    if (external_solutions and not isinstance(external_solutions[0][1], list)):
        t_inj = 0.
        if '_inj' in method:
            t_inj = 1.  # inject after cuopt's first solution generation, not instead
        external_solutions = [[t_inj, sol] for sol in external_solutions]

    sols_cars = [np.sum(np.array(sol[1][:-1]) == 0) for sol in external_solutions]
    best_cars = np.min(sols_cars) if len(sols_cars) > 0 else np.nan
    if avoid_suboptimal_cars and external_solutions:
        if USE_REF_VEHICLES:
            optimal_cars = n_carriers
        elif ENV == 'vrp':
            optimal_cars = int(np.ceil(np.sum(problems['demand'][ind]) / np.ceil(problems['capacity'][ind])))
        else:
            optimal_cars = np.min(sols_cars)
        external_solutions = [sol for sol, cars in zip(external_solutions, sols_cars)
                              if cars <= optimal_cars]

    ext_costs = [eval_utils.solution_cost(sol[1], problems['distance_matrix'][ind], LP) for sol in external_solutions]
    best_cost = np.min(ext_costs) if len(ext_costs) > 0 else np.nan
    if SORT_SOLUTIONS == 'rand':
        np.random.shuffle(external_solutions)
    elif SORT_SOLUTIONS == 'cost':
        external_solutions = [sol for _, sol in sorted(zip(ext_costs, external_solutions))]
    else:
        warnings.warn(f'Invalid sort method ({SORT_SOLUTIONS}). Keeping the current arbitrary order.')

    if 'cuOpt' not in method:
        external_solutions = [sol[1] for sol in external_solutions]
    elif SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT and external_solutions:
        external_solutions = [[t, sol[:-1]] for t, sol in external_solutions]

    return external_solutions, best_cars, best_cost, extra_time_needed


def get_naive_solutions(problems, n_solutions, problems_range, use_greedy=False, print_time=True):
    raise NotImplementedError('Generating naive solutions on-the-fly is no longer supported. '
                              'Please generate them in advance and use loading here.')
    # if print_time:
    #     t0 = time.time()
    # first = 'greedy' if use_greedy else 'random_0'
    # solvers_list = [first] + [f'random_{i:d}' for i in range(1, n_solutions)]
    # solutions = naive_solvers.get_naive_solutions(problems, solvers_list, problems_range)  # solutions[solver_name][i_problem]
    # solutions = [[sol for sol in solver_output['solutions']] for solver_output in solutions.values()]  # solutions[i_solver][i_problem]
    # solutions = [list(prob_sols) for prob_sols in zip(*solutions)]  # solutions[i_problem][i_solver]
    # if print_time:
    #     dt = time.time() - t0
    #     print(f'Naive solutions generation time: {dt:.0f}s ({dt/len(solutions):.0f}s per problem)')
    # return solutions


def convert_cuopt_solution(cuopt_solution):
    cuopt_route = cuopt_solution.get_route()
    if isinstance(cuopt_route, cudf.DataFrame):
        cuopt_route = cuopt_route.to_pandas()
    # sort by vehicles
    cuopt_route['tmp_idx'] = np.arange(len(cuopt_route))
    cuopt_route = cuopt_route.sort_values(by=['truck_id', 'tmp_idx']).drop(columns=['tmp_idx'])
    # to list
    cuopt_route = cuopt_route.route.values.tolist()
    # When changing between vehicles, depot (0) appears both as the end-point of the former and the start-point
    # of the latter. Remove such duplicated depots.
    cuopt_route = cuopt_route[:1] + [x2 for x1, x2 in zip(cuopt_route[:-1], cuopt_route[1:]) if x1 != 0 or x2 != 0]
    return cuopt_route


def get_num_carriers_candidates(tot_demand, max_capacity, margin=1):
    min_n_carriers = int(np.ceil((tot_demand / max_capacity)))
    return min_n_carriers + margin


def optimality_guarantee(sol, problems, i_problem):
    if problems['env_type'] != 'vrp':
        return False
    optimal_cars = int(np.ceil(np.sum(problems['demand']) / np.ceil(problems['capacity'])))
    optimal_cars_in_problem = optimal_cars[i_problem]
    sol_cars = np.sum(np.array(sol[0]) == 0) - 1
    if sol_cars < optimal_cars_in_problem:
        raise ValueError(i_problem, optimal_cars_in_problem, sol_cars)
    return sol_cars == optimal_cars_in_problem


def get_initialization_runtimes(config, problem_size, methods):
    RL_RUNTIME = config['injection']['rl_runtime']
    if RL_RUNTIME is None:
        # RL_RUNTIME = {50:0.008, 100:0.029, 200:0.149, 300:0.406, 400:0.845, 500:1.52}[problem_size]  # for 16 solutions
        RL_RUNTIME = {50:0.005, 100:0.016, 200:0.079, 300:0.198, 400:0.421, 500:0.767}[problem_size]  # for 8 solutions
    else:
        RL_RUNTIME = float(RL_RUNTIME)
    NAIVE_RUNTIME = config['injection']['naive_runtime']
    if NAIVE_RUNTIME is None:
        NAIVE_RUNTIME = {50:0.001, 100:0.001, 200:0.003, 300:0.008, 400:0.016, 500:0.027}[problem_size]  # for 1 solution
    else:
        NAIVE_RUNTIME = float(NAIVE_RUNTIME)
    INIT_RUNTIMES = dict(cuOpt=0, cuOpt_RL0=0, cuOpt_naive=NAIVE_RUNTIME, cuOpt_RL=RL_RUNTIME)
    for method in methods:
        if method not in INIT_RUNTIMES:
            warnings.warn(f'Init-runtime undefined for method {method}. setting to 0.')
            INIT_RUNTIMES[method] = 0

    return INIT_RUNTIMES, NAIVE_RUNTIME


def pre_stats_rlopt_vs_cuopt(problems, rlopt_solutions, LP=2):
    cuopt_time = problems['config']['cuopt']['time_limit']
    cuopt_cars = problems['baseline_vehicles']
    rlopt_cars = [np.sum(np.array(rlopt_sol[0]) == 0) - 1 if len(rlopt_sol) > 0 else np.inf
                  for rlopt_sol in rlopt_solutions]
    rlopt_extra_cars = [rl - cu for rl, cu in zip(rlopt_cars, cuopt_cars)]
    print(f'\nn_vehicles(rlopt) - n_vehicles(cuopt, {cuopt_time:d}s):', collections.Counter(rlopt_extra_cars))

    if problems['env_type'] == 'vrp':
        lower_bound_cars = int(np.ceil(np.sum(problems['demand']) / np.ceil(problems['capacity'])))
        bound_extra_cars = [rl - lb for rl, lb in zip(rlopt_cars, lower_bound_cars)]
        print(f'\nn_vehicles(rlopt) - lower_bound:', collections.Counter(bound_extra_cars))
        rlopt_optimality_guarantee = np.mean(np.array(bound_extra_cars) == 0)
        print(f'\nRLopt is guaranteed to have optimal vehicles in {100*rlopt_optimality_guarantee:.1f}% of the problems.')

    distances = problems['distance_matrix']
    cuopt_costs = problems['baseline_cost']
    rl_costs = [eval_utils.solution_cost(problem_sols[0], dist, LP) if len(problem_sols) > 0 else np.inf
                for problem_sols, dist in zip(rlopt_solutions, distances)]
    rl_mean_costs = [np.mean([eval_utils.solution_cost(sol, dist, LP) for sol in problem_sols]) if len(problem_sols) > 0 else np.inf
                     for problem_sols, dist in zip(rlopt_solutions, distances)]
    gaps = [x/y for x,y in zip(rl_costs, cuopt_costs)]
    mean_gaps = [x/y for x,y in zip(rl_mean_costs, cuopt_costs)]
    gaps_per_cars = pd.DataFrame(dict(extra_cars=rlopt_extra_cars, cost_gap=gaps)).groupby('extra_cars')
    print(f'\nCost gap: mean(best_solution)={np.mean(gaps):.3f}; mean(mean_solution)={np.mean(mean_gaps):.3f}')
    print('\nCost gap conditioned on vehicle gap:')
    print('mean(best cost gap):')
    print(gaps_per_cars.apply(lambda d: d.cost_gap.mean()))
    # print('min cost gap:')
    # print(gaps_per_cars.apply(lambda d: d.cost_gap.min()))

    utils.edge_coverage_analysis(rlopt_solutions)
    print('\n=====================')

def update_accepted_stats(solutions, distances, accepted, accepted_best, accepted_mean, accepted_cover,
                          accepted_mean_sharing, LP=2):
    # filter accepted
    if len(solutions) > 0:
        sol_is_pair = isinstance(solutions[0][1], list)
        solutions = [(sol[1] if sol_is_pair else sol) for sol, acc in zip(solutions, accepted.values) if acc==1]

    # calc stats
    if len(solutions) > 0:
        costs = [eval_utils.solution_cost(sol, distances, LP) for sol in solutions]
        best = np.min(costs)
        mean = np.mean(costs)
    else:
        best, mean = np.nan, np.nan
    if len(solutions) > 1:
        rel_cover, mean_shared_edges = utils.edge_coverage_analysis([solutions], verbose=0)
    else:
        rel_cover, mean_shared_edges = np.nan, np.nan

    # update stats
    accepted_best.append(best)
    accepted_mean.append(mean)
    accepted_cover.append(rel_cover)
    accepted_mean_sharing.append(mean_shared_edges)

def get_pools_bests(populations):
    times, pool_sizes, costs = [], [], []
    for t, population_costs, population_valids in zip(
            populations.times, populations.snapshot_costs, populations.valid_snapshots):
        population_costs = [c for c, v in zip(population_costs, population_valids) if v]
        times.append(t)
        pool_sizes.append(len(population_costs))
        costs.append(np.min(population_costs) if len(population_costs)>0 else None)
    return times, pool_sizes, costs

def summarize_results(res, pools, info, out_path, allow_wandb, baseline, main_method, methods, verbose=0):
    # save results summary
    try:
        rr = pd.DataFrame(res)
    except:
        print({k: len(v) for k, v in res.items()})
        raise

    # for each method and time budget, mark the method as relevant if it has >=50% success
    def is_valid_method(d, alpha=0.5):
        d['is_relevant_method'] = d.success.mean() >= alpha
        return d
    rr = rr.groupby(['method', 'total_runtime']).apply(lambda d: is_valid_method(d)).reset_index(drop=True)
    # for each point of comparison (time budget, problem): check if all relevant methods have a feasible solution
    def all_methods_feasible(d):
        d['all_methods_feasible'] = d[d.is_relevant_method].success.all()
        return d
    rr = rr.groupby(['total_runtime', 'problem_id']).apply(all_methods_feasible).reset_index(drop=True)

    try:
        pools_res = pd.DataFrame(pools)
    except:
        print({k: len(v) for k, v in pools.items()})
        raise

    print(f'Saving results to {out_path}')
    with open(out_path, 'wb') as hh:
        pkl.dump(dict(summary=rr, pools=pools_res, info=info), hh)

    if allow_wandb:
        wandb.save(out_path, base_path=wandb.run.dir)

    # print results
    if verbose >= 1:
        with pd.option_context('display.max_columns', None, 'display.width', 0):
            print(rr)

    if verbose >= 2:
        print()
        print(pools_res)

    # compare main to baseline
    def print_mean_cost(r, cost_col):
        mean = r[cost_col].mean()
        std = r[cost_col].std()
        n = len(r)
        return f'{mean:.4f} +- {std/np.sqrt(n):.4f} (n={n:d})'
    def print_pairwise_comparison(r, cost_col, method1, method2):
        cost_gap = r[r.method==method2][cost_col].values - r[r.method==method1][cost_col].values
        mean = np.mean(cost_gap)
        std = np.std(cost_gap)
        n = len(cost_gap)
        return f'{cost_col}: {method2}-{method1} = {mean:.4f} +- {std/np.sqrt(n):.4f} (n={n:d})'

    print('\nFeasible solutions:')
    print(rr.groupby(['total_runtime', 'method']).apply(
        lambda r: print_mean_cost(r, 'success')))#, include_groups=False))

    rf = rr[rr.all_methods_feasible]

    print('\nVehicles summary (only counting feasible solutions):')
    print(rf.groupby(['total_runtime', 'method']).apply(
        lambda r: print_mean_cost(r, 'vehicles')))#, include_groups=False))
    if baseline in methods and main_method in methods:
        print(rf.groupby(['total_runtime']).apply(lambda r: print_pairwise_comparison(
            r, 'vehicles', baseline, main_method)))#, include_groups=False))

    print('\nCost summary (only counting feasible solutions):')
    print(rf.groupby(['total_runtime', 'method']).apply(
        lambda r: print_mean_cost(r, 'cost')))#, include_groups=False))
    if baseline in methods and main_method in methods:
        print(rf.groupby(['total_runtime']).apply(lambda r: print_pairwise_comparison(
            r, 'cost', baseline, main_method)))#, include_groups=False))
    else:
        warnings.warn(f'Baseline {baseline} and/or Main method {main_method} are not available '
                      f'for pairwise comparison analysis.')

    return rr, pools_res

def run_initialization_only(external_solutions, best_cars, best_cost, method):
    success = len(external_solutions) > 0
    vehicle_count = best_cars
    cost = best_cost
    solution = external_solutions[0] if len(external_solutions) else []
    if len(solution) > 0 and 'cuOpt' in method:
        solution = solution[1] + [0]
    accepted = pd.Series(dtype='float')
    actual_runtime = 0
    num_iterations = 0
    return success, vehicle_count, cost, solution, accepted, actual_runtime, num_iterations

def run_cuopt(external_solutions, problems, demands_for_solver, n_carriers, ind, ga_time, ENV, LAST_RETURN_TO_DEPOT, FIXED_VEHICLES, PULL_FREQ, CUOPT_NORMALIZATION):
    # construct cuopt solver
    cuopt_config = dict(
        problem_setup=dict(
            env=ENV,
            last_return_to_depot=LAST_RETURN_TO_DEPOT
        ),
        cuopt=dict(
            time_limit=ga_time,
            climbers=2048,
            pull_frequency=PULL_FREQ,
        ),
    )
    solver = CuOptSolver(cuopt_config)
    # print('Time limit:', solver.solver_settings.get_time_limit(), '\n')

    distance_matrix = problems['distance_matrix'][ind]
    capacity = problems['capacity'][ind]
    problem = solver.create_data_model(
        demand=demands_for_solver,
        distance_matrix=distance_matrix / CUOPT_NORMALIZATION,
        vehicle_capacity=np.full(shape=(n_carriers,),
                                 fill_value=capacity,
                                 dtype=int),
        n_carriers=n_carriers, set_fixed_carriers=FIXED_VEHICLES,
        last_drop_return_trip=not cuopt_config['problem_setup']['last_return_to_depot'],
        injected_solutions=external_solutions)

    # run cuopt solver
    t0 = time.time()
    try:
        routing_solution, populations = solver.solve(problem, verbose=0)
        actual_runtime = time.time() - t0
        success = routing_solution.get_status() == 0
        vehicle_count = routing_solution.get_vehicle_count()
        cost = CUOPT_NORMALIZATION * routing_solution.get_total_objective()
        solution = convert_cuopt_solution(routing_solution)
        accepted = routing_solution.get_accepted_solutions()
    except Exception as e:
        warnings.warn(f"Caught a failure in cuOpt: {e}")
        actual_runtime = time.time() - t0
        success, vehicle_count, cost, solution = False, np.nan, np.nan, []
        accepted = pd.Series(dtype='float')
        populations = None

    num_iterations = np.nan

    if populations is not None:
        populations.update_costs(distance_matrix=distance_matrix, capacity=capacity,
                                 demands_for_solver=demands_for_solver, n_customers=distance_matrix.shape[0] - 1,
                                 return_to_depot=cuopt_config['problem_setup']['last_return_to_depot'])

    return success, vehicle_count, cost, solution, accepted, actual_runtime, num_iterations, populations


def main(config_path='config_initialization.yaml'):
    # configuration
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    logger = UnifiedLogger(config)
    # wandb_mode = 'online' if config['system']['allow_wandb'] else 'disabled'
    wandb_mode = 'disabled'
    wandb.init(mode=wandb_mode)

    RUN_TITLE = config['names']['run_title']
    PROBLEMS_PATH = config['paths']['problems']
    SOLUTION_PATHS = config['paths']['solutions']
    if isinstance(SOLUTION_PATHS, str):
        SOLUTION_PATHS = [SOLUTION_PATHS]
    SOLUTION_SOURCES = config['paths']['solutions_source']
    if SOLUTION_SOURCES is not None and "(" in SOLUTION_SOURCES:
        assert SOLUTION_SOURCES[0]=="(" and SOLUTION_SOURCES[-1]==")"
        SOLUTION_SOURCES = SOLUTION_SOURCES[1:-1].replace(' ','').split(',')
    if not isinstance(SOLUTION_SOURCES, (tuple, list)):
        SOLUTION_SOURCES = [SOLUTION_SOURCES]
    NAIVE_SOURCE = config['paths']['naive_source']
    PROBLEMS_RANGE = config['problem']['problems_range']
    replacement = ''
    if PROBLEMS_RANGE is not None:
        PROBLEMS_RANGE = [int(idx) for idx in PROBLEMS_RANGE]
        replacement += f'_{PROBLEMS_RANGE[0]:d}-{PROBLEMS_RANGE[1]-1:d}'
    RUN_TITLE = RUN_TITLE.replace('PLACEHOLDER', replacement)

    OUT_PATH = f'{config["paths"]["out_summary"]}_{RUN_TITLE}.pkl'
    OUT_PATH = os.path.join('./outputs', OUT_PATH)
    VERBOSE = config['names']['verbose']

    METHODS = config['names']['methods']
    BASELINE = config['names']['baseline']
    MAIN_METHOD = config['names']['main_method']
    SOLVER_RUNTIMES = [float(x) for x in config['cuopt']['runtimes']]
    if BASELINE not in METHODS:
        BASELINE = None
    if MAIN_METHOD not in METHODS:
        MAIN_METHOD = None

    MAX_EXTERNAL_SOLUTIONS = config['injection']['max_injections']
    if MAX_EXTERNAL_SOLUTIONS is not None:
        if isinstance(MAX_EXTERNAL_SOLUTIONS, (tuple, list)):
            MAX_EXTERNAL_SOLUTIONS = [int(n) for n in MAX_EXTERNAL_SOLUTIONS]
        else:
            MAX_EXTERNAL_SOLUTIONS = int(MAX_EXTERNAL_SOLUTIONS)
    CHOOSE_EXTERNAL_SOLUTIONS = config['injection']['can_choose_injections']
    BOTH_RL_AND_NAIVE = config['injection']['both_rl_and_naive']
    N_NAIVE_SOLUTIONS = config['injection']['n_naives']
    SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT = config['problem']['remove_last_depot_from_solution']
    AVOID_SUBOPTIMAL_CARS = config['injection']['avoid_suboptimal']
    SORT_SOLUTIONS = config['injection']['sort']
    REPETITIONS = config['cuopt']['repetitions']

    ENV = config['problem']['env']
    LAST_RETURN_TO_DEPOT = config['problem']['last_return_to_depot']
    FIXED_VEHICLES = config['cuopt']['fixed_vehicles']
    SPARE_VEHICLES = config['cuopt']['spare_vehicles']
    if SPARE_VEHICLES > 0:
        FIXED_VEHICLES = False
    USE_REF_VEHICLES = config['cuopt']['use_reference_for_num_vehicles']
    PULL_FREQ = config['cuopt']['pull_freq']

    print(f'\n{RUN_TITLE}')
    print('CuOpt version:', cuopt.__version__)
    print('Problems:', PROBLEMS_PATH)
    delete_solutions = False
    if SOLUTION_SOURCES[0] is None:
        print('Solutions:', SOLUTION_PATHS)
    else:
        print('Solutions sources:', SOLUTION_SOURCES)
        SOLUTION_PATHS = [download_wandb_file(run_id, config['system']['wandb_user'], config['system']['wandb_project'])
                          for run_id in SOLUTION_SOURCES]
        delete_solutions = True

    # load problems
    problems, problems_range, problem_size, n_problems, CUOPT_NORMALIZATION, LP = load_problems(
        PROBLEMS_PATH, PROBLEMS_RANGE, config)
    log10_problems = int(np.ceil(np.log10(n_problems)))

    # get initialization runtimes
    INIT_RUNTIMES, NAIVE_RUNTIME = get_initialization_runtimes(config, problem_size, METHODS)

    # get initial solutions
    rlopt_solutions, rlopt_vehicles, rlopt_costs, naive_solutions = get_initial_solutions(
        config, problems, problems_range, METHODS, SOLUTION_PATHS, NAIVE_SOURCE, MAX_EXTERNAL_SOLUTIONS,
        CHOOSE_EXTERNAL_SOLUTIONS, delete_solutions, LP)

    # Initial stats on rlopt vs cuopt's original solutions
    if VERBOSE >= 3 and rlopt_solutions:
        pre_stats_rlopt_vs_cuopt(problems, list(rlopt_solutions.values())[0], LP)

    # initialize results columns
    methods = []
    solver_times = []
    init_times = []
    problem_ids = []
    vehicles = []
    costs = []
    n_external = []
    n_accepted = []
    n_rejected = []
    n_unconsidered = []
    accepted_best = []
    accepted_mean = []
    accepted_cover = []
    accepted_mean_sharing = []
    extra_naive_injection = []
    solver_iterations = []
    solutions = {}
    stats = {}
    successes = []
    successes2 = []
    runtimes = []

    pool_methods, pool_full_times, pool_times, pool_problems, pool_sizes, pool_vehicles, pool_costs = \
        [], [], [], [], [], [], []

    # run test
    for i_time, solver_time in enumerate(SOLVER_RUNTIMES):
        print(f'\nSOLVER_RUNTIME = {solver_time}:')
        for ind in range(*problems_range):
            for i_method, method in enumerate(METHODS):
                # set cuopt configuration
                init_runtime = INIT_RUNTIMES[method]
                ga_time = solver_time - init_runtime

                # extract instance data
                demands_for_solver = problems['demand'][ind]
                tot_demand = np.sum(demands_for_solver)
                max_capacity = problems['capacity'][ind]
                if USE_REF_VEHICLES:
                    n_carriers = problems['baseline_vehicles'][ind].item() + SPARE_VEHICLES
                else:
                    n_carriers = get_num_carriers_candidates(
                        tot_demand, max_capacity, margin=0 if FIXED_VEHICLES else SPARE_VEHICLES)

                avoid_suboptimal_cars = AVOID_SUBOPTIMAL_CARS
                external_solutions, best_cars, best_cost, extra_time_needed = preprocess_external_solutions(
                    method, ind, problems, n_carriers, rlopt_solutions, naive_solutions, problems_range, avoid_suboptimal_cars=avoid_suboptimal_cars, force_feasible=ga_time == 0, LP=LP,
                    BOTH_RL_AND_NAIVE=BOTH_RL_AND_NAIVE, NAIVE_RUNTIME=NAIVE_RUNTIME, USE_REF_VEHICLES=USE_REF_VEHICLES, ENV=ENV, SORT_SOLUTIONS=SORT_SOLUTIONS,
                    SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT=SOLUTIONS_FORMAT_HAS_NO_LAST_DEPOT)
                if extra_time_needed > 0:
                    init_runtime += extra_time_needed
                    ga_time = max(ga_time - extra_time_needed, 0)

                for repetition in range(REPETITIONS):

                    solver_stats = None

                    if ('naive' in method or 'RL' in method) and ga_time == 0:
                        # CASE I: initialization only
                        success, vehicle_count, cost, solution, accepted, actual_runtime, num_iterations = run_initialization_only(
                            external_solutions, best_cars, best_cost, method)

                    else:
                        # CASE II: cuOpt
                        success, vehicle_count, cost, solution, accepted, actual_runtime, num_iterations, populations = \
                            run_cuopt(external_solutions, problems, demands_for_solver, n_carriers, ind, ga_time, ENV, LAST_RETURN_TO_DEPOT, FIXED_VEHICLES, PULL_FREQ, CUOPT_NORMALIZATION)

                    # re-verify solution
                    if success:
                        success2 = eval_utils.verify_solution(
                            solution, demands_for_solver, max_capacity.item(), n_carriers, True, print)
                        if not success2:
                            print(f'Solution validation failed ({solver_time}, {ind}, {method}).')
                    else:
                        # if first indicator is failure - don't trust verifier to work properly, and default to False.
                        try:
                            success2 = eval_utils.verify_solution(
                                solution, demands_for_solver, max_capacity.item(), n_carriers, False)
                        except:
                            success2 = False

                    # update results
                    problem_id = ind if REPETITIONS == 1 else str(ind)+chr(97+repetition)
                    methods.append(method)
                    solver_times.append(solver_time)
                    init_times.append(init_runtime)
                    runtimes.append(init_runtime + actual_runtime)
                    problem_ids.append(problem_id)
                    successes.append(success)
                    successes2.append(success2)
                    vehicles.append(vehicle_count)
                    costs.append(cost)
                    n_external.append(len(accepted))
                    n_accepted.append((accepted == 1).astype(int).sum())
                    n_rejected.append((accepted == 0).astype(int).sum())
                    n_unconsidered.append((accepted == -1).astype(int).sum())
                    extra_naive_injection.append(extra_time_needed > 0)
                    solver_iterations.append(num_iterations)
                    update_accepted_stats(external_solutions, problems['distance_matrix'][ind], accepted,
                                          accepted_best, accepted_mean, accepted_cover, accepted_mean_sharing, LP)
                    solutions[f'time_{solver_time:.0f}_problem_{ind:d}_{method:s}_iter_{repetition:d}'] = solution
                    if solver_stats is not None:
                        stats[f'time_{solver_time:.0f}_problem_{ind:d}_{method:s}_iter_{repetition:d}'] = solver_stats

                    # update pool results
                    if PULL_FREQ > 0 and method.startswith('cuOpt'):
                        tmp = get_pools_bests(populations)
                        n = len(tmp[0])
                        pool_times.extend(tmp[0])
                        pool_sizes.extend(tmp[1])
                        pool_costs.extend(tmp[2])
                        pool_methods.extend(n*[method])
                        pool_full_times.extend(n*[solver_time])
                        pool_problems.extend(n*[problem_id])
                        pool_vehicles.extend(n*[None])

                print(f'Finished problem {i_time+1:d}.{ind-problems_range[0]+1:0{log10_problems}d}.{i_method+1:d}/'
                      f'{len(SOLVER_RUNTIMES):d}.{n_problems:0{log10_problems}d}.{len(METHODS):d}.')

        summary_verbosity = VERBOSE if i_time == len(SOLVER_RUNTIMES) - 1 else 0
        res = dict(
            method=methods, total_runtime=solver_times, init_runtime=init_times, problem_id=problem_ids,
            vehicles=vehicles, cost=costs, attempted_injections=n_external, accepted=n_accepted, rejected=n_rejected,
            unconsidered=n_unconsidered, accepted_best=accepted_best, accepted_mean=accepted_mean,
            accepted_cover=accepted_cover, accepted_mean_sharing=accepted_mean_sharing,
            extra_naive_injection=extra_naive_injection, solver_iterations=solver_iterations,
            success=successes, success_verification=successes2,
        )
        pools = dict(
            method=pool_methods, final_time=pool_full_times, problem_id=pool_problems, time=pool_times,
            pool_size=pool_sizes, vehicles=pool_vehicles, best_cost=pool_costs,
        )
        info = dict(
            config=config,
            external_costs=dict(vehicles=rlopt_vehicles, costs=rlopt_costs),
            solutions=solutions, stats=stats,
            cuopt_version=cuopt.__version__,
        )
        rr, pools_res = summarize_results(res, pools, info, OUT_PATH, config['system']['allow_wandb'], BASELINE, MAIN_METHOD, METHODS, summary_verbosity)

if __name__ == '__main__':
    main()
