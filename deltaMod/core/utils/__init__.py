import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dates import Dates

log = logging.getLogger(__name__)



def set_system_spec(cuda_devices: Optional[list] = None) -> Tuple[str, str]:
    """Set the device and data type for the model on user's system.

    Parameters
    ----------
    cuda_devices : list
        List of CUDA devices to use. If None, the first available device is used.

    Returns
    -------
    Tuple[str, str]
        The device type and data type for the model.
    """
    if cuda_devices != []:
        # Set the first device as the active device.
        # d = cuda_devices[0]
        if torch.cuda.is_available() and cuda_devices < torch.cuda.device_count():
            device = torch.device(f'cuda:{cuda_devices}')
            torch.cuda.set_device(device)   # Set as active device.
        else:
            raise ValueError(f"Selected CUDA device {cuda_devices} is not available.")  
    
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        torch.cuda.set_device(device)   # Set as active device.

    elif torch.backends.mps.is_available():
        # Use Mac M-series ARM architecture.
        device = torch.device('mps')

    else:
        device = torch.device('cpu')
    
    dtype = torch.float32
    return str(device.type), str(dtype)


def initialize_config(config: Union[DictConfig, dict]) -> Dict[str, Any]:
    """Parse and initialize configuration settings.
    
    Parameters
    ----------
    config : DictConfig
        Configuration settings from Hydra.
        
    Returns
    -------
    dict
        Formatted configuration settings.
    """
    if type(config) == DictConfig:
        try:
            config = OmegaConf.to_container(config, resolve=True)
        except ValidationError as e:
            log.exception("Configuration validation error", exc_info=e)
            raise e
    
    
    config['device'], config['dtype'] = set_system_spec(config['gpu_id'])

    # Convert date ranges to integer values.
    config['train_t_range'] = Dates(config['train'], config['dpl_model']['rho']).date_to_int()
    config['test_t_range'] = Dates(config['test'], config['dpl_model']['rho']).date_to_int()
    config['total_t_range'] = [config['train_t_range'][0], config['test_t_range'][1]]
    
    # Create output directories.
    config = create_output_dirs(config)

    return config


def set_randomseed(seed=0) -> None:
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed to set. If None, a random seed is generated (default 0).
    """
    if seed == None:
        randomseed = int(np.random.uniform(low=0, high=1e6))
        pass

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True)
    except:
        pass
    

def create_output_dirs(config: Dict[str, Any]) -> dict:
    """Create output directories for saving models and results.

    Parameters
    ----------
    config : dict
        Configuration dictionary with paths and model settings.
    
    Returns
    -------
    dict
        The original config with path modifications.
    """
    # Add dir for train period:
    train_period = 'train_' + str(config['train']['start_time'][:4]) + '_' +  \
        str(config['train']['end_time'][:4])

    # Add dir for number of forcings:
    forcings = str(len(config['nn_forcings'])) + '_forcing'

    # Add dir for ensemble type:
    if config['ensemble_type'] in ['none', '']:
        ensemble_state = 'no_ensemble'
    else:
        ensemble_state = config['ensemble_type']
    
    # Add dir for:
    #  1. model name(s)
    #  2. static or dynamic parametrization
    #  3. loss functions per model.
    mod_names = ''
    dy_params = ''
    loss_fn = ''
    for mod in config['phy_model']['models']:
        mod_names += mod + '_'

        for param in config['phy_model']['dy_params'][mod]:
            dy_params += param + '_'
        
        loss_fn += config['loss_function']['model'] + '_'

    # Add dir for hyperparam spec.
    params = config['pnn_model']['model'] + \
             '_E' + str(config['train']['epochs']) + \
             '_R' + str(config['dpl_model']['rho'])  + \
             '_B' + str(config['train']['batch_size']) + \
             '_H' + str(config['pnn_model']['hidden_size']) + \
             '_n' + str(config['dpl_model']['nmul']) + \
             '_' + str(config['random_seed']) 

    # If any model in ensemble is dynamic, whole ensemble is dynamic.
    dy_state = 'static_para' if dy_params.replace('_','') == '' else 'dynamic_para'
    
    # ---- Combine all dirs ---- #
    model_path = os.path.join(config['save_path'],
                              config['observations']['name'],
                              train_period,
                              forcings,
                              ensemble_state,
                              params,
                              mod_names,
                              loss_fn,
                              dy_state)

    if dy_state == 'dynamic_para':
        model_path = os.path.join(model_path, dy_params)

    test_period = 'test' + str(config['test']['start_time'][:4]) + '_' + \
        str(config['test']['end_time'][:4])
    test_path = os.path.join(model_path, test_period)

    # Create the directories.
    if (config['mode'] == 'test') and (os.path.exists(model_path) == False):
        if config['ensemble_type'] in ['avg', 'frozen_pnn']:
            for mod in config['phy_model']['models']:
                # Check if individually trained models exist and use those.
                check_path = os.path.join(config['save_path'],
                                          config['observations']['name'],
                                          train_period,
                                          forcings,
                                          'no_ensemble',
                                          params,
                                          dy_state,
                                          mod + '_')
                if os.path.exists(check_path) == False:           
                    raise FileNotFoundError(f"Attempted to test with individually trained models but {check_path} not found. Check config or train models before testing.")
        else:
            raise FileNotFoundError(f"Model directory {model_path} not found. Check config or train models before testing.")

    # Create the directories if they don't exist.
    os.makedirs(test_path, exist_ok=True)
    
    # Saving the config file to output path (overwrite if exists).
    config_file = json.dumps(config)
    config_path = os.path.join(model_path, 'config_file.json')
    if os.path.exists(config_path):
        os.remove(config_path)
    with open(config_path, 'w') as f:
        f.write(config_file)

    # Append the output directories to the config.
    config['out_path'] = model_path
    config['testing_path'] = test_path
    
    return config


def save_model(config, model, model_name, epoch, create_dirs=False) -> None:
    """
    Save ensemble or single models.
    """
    # If the model folder has not been created, do it here.
    if create_dirs: create_output_dirs(config)

    save_name = str(model_name) + '_model_Ep' + str(epoch) + '.pt'
    # os.makedirs(save_name, exist_ok=True)

    full_path = os.path.join(config['output_dir'], save_name)
    torch.save(model, full_path)


def save_outputs(config, preds_list, y_obs, create_dirs=False) -> None:
    """
    Save outputs from a model.
    """
    if create_dirs: create_output_dirs(config)

    for key in preds_list[0].keys():
        if config['ensemble_type'] != 'none':
            if len(preds_list[0][key].shape) == 3:
                dim = 0
            else:
                dim = 1
        else:
            if len(preds_list[0][key].shape) == 3:
                dim = 1
            else:
                dim = 0

        concatenated_tensor = torch.cat([d[key] for d in preds_list], dim=dim)
        file_name = key + ".npy"        

        np.save(os.path.join(config['testing_dir'], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in config['target']:
        item_obs = y_obs[:, :, config['target'].index(var)]
        file_name = var + '.npy'
        np.save(os.path.join(config['testing_dir'], file_name), item_obs)


def load_model(config, model_name, epoch):
    """Load trained PyTorch models.
    
    Args:
        config (dict): Configuration dictionary with paths and model settings.
        model_name (str): Name of the model to load.
        epoch (int): Epoch number to load the specific state of the model.
        
    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    model_name = str(model_name) + '_model_Ep' + str(epoch) + '.pt'
    # model_path = os.path.join(config['output_dir'], model_name)
    # try:
    #     self.model_dict[model] = torch.load(model_path).to(self.config['device']) 
    # except:
    #     raise FileNotFoundError(f"Model file {model_path} was not found. Check that epochs and hydro models in your config are correct.")

    # # Construct the path where the model is saved
    # model_file_name = f"{model_name}_epoch_{epoch}.pth"
    # model_path = os.path.join(config['model_dir'], model_file_name)
    
    # # Ensure the model file exists
    # if not os.path.isfile(model_path):
    #     raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    # return torch.load(model_path)
    
    # Retrieve the model class from config (assuming it's stored in the config)
    # model_class = config['model_classes'][model_name]
    
    # Initialize the model (assumes model classes are callable and take no arguments)
    # model = model_class()
    # Load the state_dict into the model
    # model.load_state_dict(state_dict)
    
    # return model


def print_config(config: Dict[str, Any]) -> None:
    """Print the current configuration settings.

    Parameters
    ----------
    config : dict
        Dictionary of configuration settings.

    Adapted from: Jiangtao Liu
    """
    print()
    print("\033[1m" + "Current Configuration" + "\033[0m")
    print(f"  {'Experiment Mode:':<20}{config['mode']:<20}")
    print(f"  {'Ensemble Mode:':<20}{config['ensemble_type']:<20}")

    for i, mod in enumerate(config['phy_model']['models']):
        print(f"  {f'Model {i+1}:':<20}{mod:<20}")
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f"  {'Data Source:':<20}{config['observations']['name']:<20}")
    if config['mode'] != 'test':
        print(f"  {'Train Range :':<20}{config['train']['start_time']:<20}{config['train']['end_time']:<20}")
    if config['mode'] != 'train':
        print(f"  {'Test Range :':<20}{config['test']['start_time']:<20}{config['test']['end_time']:<20}")
    if config['train']['run_from_checkpoint'] == True:
        print(f"  {'Resuming training from epoch:':<20}{config['run_from_checkpoint']['start_epoch']:<20}")
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f"  {'Train Epochs:':<20}{config['train']['epochs']:<20}{'Batch Size:':<20}{config['train']['batch_size']:<20}")
    print(f"  {'Dropout:':<20}{config['pnn_model']['dropout']:<20}{'Hidden Size:':<20}{config['pnn_model']['hidden_size']:<20}")
    print(f"  {'Warmup:':<20}{config['phy_model']['warm_up']:<20}{'Concurrent Models:':<20}{config['dpl_model']['nmul']:<20}")
    print(f"  {'Optimizer:':<20}{config['loss_function']['model']:<20}")
    print()

    # if 'pnn' in config['ensemble_type']:
    #     print("\033[1m" + "Weighting Network Parameters" + "\033[0m")
    #     print(f'  {"Dropout:":<20}{config.weighting_nn.dropout:<20}{"Hidden Size:":<20}{config.weighting_nn.hidden_size:<20}')
    #     print(f'  {"Method:":<20}{config.weighting_nn.method:<20}{"Loss Factor:":<20}{config.weighting_nn.loss_factor:<20}')
    #     print(f'  {"Loss Lower Bound:":<20}{config.weighting_nn.loss_lower_bound:<20}{"Loss Upper Bound:":<20}{config.weighting_nn.loss_upper_bound:<20}')
    #     print(f'  {"Optimizer:":<20}{config.weighting_nn.loss_function:<20}')
    #     print()

    print("\033[1m" + "Machine" + "\033[0m")
    print(f"  {'Use Device:':<20}{str(config['device']):<20}")
    print()
    