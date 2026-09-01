# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import os
import torch
import numpy as np

from .utils.safe_serialization import safe_pickle_load

# Removed: shutil, traceback, uuid, warnings, defaultdict, cudf, wandb, cuOpt imports
# Removed: DEFAULT_FIX_CAPACITY, DEFAULT_CAPACITIES, DEPOT_LOCATION, RESET_LOCATION, UNMASKED_INDEX
# Removed: write_dataset (function)

class ProblemLoader(object):
    """
    Loads pre-generated problem data from a pickle file.
    """
    def __init__(self, config):
        # This __init__ is not strictly necessary if all methods are static,
        # but kept for consistency or future non-static helpers.
        # self.config = config
        pass

    @staticmethod
    def load_problem_data(config, fname: str, problem_range: list = None):
        """
        Loads problem data from a .pkl file.

        Args:
            config: The main configuration dictionary (currently not used by this static method
                    but kept for API consistency or future use).
            fname: Path to the .pkl file containing the problem data.

        Returns:
            A dictionary containing the loaded and processed problem data.
        """
        if not os.path.exists(fname):
            logging.error(f'Data file {fname} not found.')
            raise FileNotFoundError(f'Data file {fname} not found.')

        logging.info(f'Loading data from {fname}')
        with open(fname, 'rb') as f:
            loaded_data_pkl = safe_pickle_load(f)

        data = {}

        # 1. env_type (String)
        data['env_type'] = loaded_data_pkl.get('env_type', 'vrp')

        # 2. positions (torch.Tensor)
        if 'positions' in loaded_data_pkl:
            pos_val = loaded_data_pkl['positions']
            if problem_range is not None:
                pos_val = pos_val[problem_range[0]:problem_range[1]]
            if isinstance(pos_val, np.ndarray):
                data['positions'] = torch.from_numpy(pos_val)
            elif isinstance(pos_val, torch.Tensor):
                data['positions'] = pos_val
            else:
                raise TypeError(f"Field 'positions' has unsupported type: {type(pos_val)}")
        else:
            raise KeyError("Field 'positions' is missing from the data file.")

        # 3. demand (torch.Tensor) - handles renaming from 'demands'
        demand_val = None
        if 'demand' in loaded_data_pkl:
            demand_val = loaded_data_pkl['demand']
        elif 'demands' in loaded_data_pkl:
            demand_val = loaded_data_pkl['demands']
        
        if demand_val is not None:
            if problem_range is not None:
                demand_val = demand_val[problem_range[0]:problem_range[1]]
            if isinstance(demand_val, np.ndarray):
                data['demand'] = torch.from_numpy(demand_val)
            elif isinstance(demand_val, torch.Tensor):
                data['demand'] = demand_val
            else:
                raise TypeError(f"Field 'demand'/'demands' has unsupported type: {type(demand_val)}")
        else:
            raise KeyError("Field 'demand' or 'demands' is missing from the data file.")

        # 4. distance_matrix (torch.Tensor)
        if 'distance_matrix' in loaded_data_pkl:
            dm_val = loaded_data_pkl['distance_matrix']
            if problem_range is not None:
                dm_val = dm_val[problem_range[0]:problem_range[1]]
            if isinstance(dm_val, np.ndarray):
                data['distance_matrix'] = torch.from_numpy(dm_val)
            elif isinstance(dm_val, torch.Tensor):
                data['distance_matrix'] = dm_val
            else:
                raise TypeError(f"Field 'distance_matrix' has unsupported type: {type(dm_val)}")
        else:
            # If distance_matrix is optional and can be computed from positions,
            # this could be a warning and a computation step.
            # For now, assuming it's required based on typical VRP datasets.
            raise KeyError("Field 'distance_matrix' is missing from the data file.")

        # 5. capacity (torch.Tensor, float) - handles renaming from 'capacities'
        capacity_val = None
        if 'capacity' in loaded_data_pkl:
            capacity_val = loaded_data_pkl['capacity']
        elif 'capacities' in loaded_data_pkl:
            capacity_val = loaded_data_pkl['capacities']

        if capacity_val is not None:
            if problem_range is not None:
                capacity_val = capacity_val[problem_range[0]:problem_range[1]]
            if isinstance(capacity_val, np.ndarray):
                data['capacity'] = torch.from_numpy(capacity_val).float()
            elif isinstance(capacity_val, torch.Tensor):
                data['capacity'] = capacity_val.float()
            elif isinstance(capacity_val, (int, float)): # Handle scalar capacity
                data['capacity'] = torch.tensor([capacity_val] * data['positions'].shape[0], dtype=torch.float32) \
                                   if data['positions'].shape[0] > 1 and np.isscalar(capacity_val) \
                                   else torch.tensor(capacity_val, dtype=torch.float32)
            else:
                raise TypeError(f"Field 'capacity'/'capacities' has unsupported type: {type(capacity_val)}")
        else:
            raise KeyError("Field 'capacity' or 'capacities' is missing from the data file.")

        # Derived and other important fields for vehicle_routing.py
        data['n_problems'] = data['positions'].shape[0]

        if 'radius' in loaded_data_pkl:
            radius_val = loaded_data_pkl['radius']
        else:
            radius_val = data['positions'].abs().max().item()
        data['radius'] = radius_val # Keep as float/int as per original direct usage

        if 'id' in loaded_data_pkl:
            ids_val = loaded_data_pkl['id']
            if isinstance(ids_val, (np.ndarray, list)):
                data['id'] = np.array(ids_val) # vehicle_routing.py expects numpy array for ids
            else:
                logging.warning(f"Field 'id' has an unsupported type: {type(ids_val)}. Skipping.")
        else:
            # Generate default IDs if missing, as RoutingBase expects self.ids
            data['id'] = np.arange(data['n_problems'])
            logging.info(f"Field 'id' missing. Generated default IDs: {data['id']}")


        # Load original problem config if present (for consistency checks in vehicle_routing.py)
        if 'config' in loaded_data_pkl:
            data['config'] = loaded_data_pkl['config']

        # Load dataset_type if present (used for consistency check in vehicle_routing.py as data['env_type'])
        # This might be redundant if loaded_data_pkl['env_type'] is already used for data['env_type']
        if 'dataset_type' in loaded_data_pkl and 'env_type' not in loaded_data_pkl:
            # If original pkl had 'dataset_type' but not 'env_type', use it for 'env_type'.
            data['env_type'] = loaded_data_pkl['dataset_type']


        # Ensure essential data used by RoutingBase.set_problem_data is present
        for key_check in ['positions', 'demand', 'distance_matrix', 'capacity', 'id', 'n_problems', 'radius', 'env_type']:
            if key_check not in data:
                # Some like 'radius' or 'id' might have logged warnings but are critical
                if key_check == 'radius' and key_check not in data: # If radius was skipped due to type
                     raise KeyError(f"Field '{key_check}' is missing or has an unsupported type and is required.")
                # id is now defaulted if missing
                # Others would have raised KeyError already if missing mandatory field from pkl
                pass
        return data

# Removed: load_real_data, generate_data, sample_real_distances, generate_distances, generate_demands,
#          generate_time_windows, create_dataset, is_better
# Removed: get_num_carriers_candidates, convert_cuopt_solution, test_cuopt_solution
# Removed global constants related to data generation
