from typing import Dict, Optional

import numpy as np
import torch
import copy

from core.data import randomIndex_percentage
from core.data.data_samplers.base import BaseDataSampler


class HydroDataSampler(BaseDataSampler):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['dpl_model']['phy_model']['warm_up']
        self.rho = config['dpl_model']['rho']

    def load_data(self):
        """Custom implementation for loading data."""
        print("Loading data...")

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        print("Preprocessing data...")

    def select_subset(
            self,
            x: torch.Tensor,
            i_grid: np.ndarray,
            i_t: Optional[np.ndarray] = None,
            c: Optional[np.ndarray] = None,
            tuple_out: bool = False,
            has_grad: bool = False
    ) -> torch.Tensor:
        """Select a subset of input tensor."""
        if isinstance(x, list):
            batch_size, nx = len(i_grid),  x[0].shape[-1]
        else:
            batch_size, nx, nt = len(i_grid), x.shape[-1], x.shape[0]
        rho, warm_up = self.rho, self.warm_up

        # Handle time indexing and create an empty tensor for selection
        if i_t is not None:
            if isinstance(x, list):  # to handle forcing data with mutliple dims
                out_list = []
                for k in range(batch_size):
                    temp = x[i_grid[k]][i_t[k] - warm_up:i_t[k] + rho]
                    out_list.append(temp)
                x_tensor = torch.stack(out_list, dim=1)
                if has_grad:
                    x_tensor.requires_grad_(True)
            else:
                x_tensor = torch.zeros(
                    [rho + warm_up, batch_size, nx],
                    device=self.device,
                    requires_grad=has_grad
                )
                for k in range(batch_size):
                    x_tensor[:, k:k + 1, :] = x[i_t[k] - warm_up:i_t[k] + rho, i_grid[k]:i_grid[k] + 1, :]
        else:
            x_tensor = x[:, i_grid, :].float().to(self.device) if x.ndim == 3 else x[i_grid, :].float().to(self.device)

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, rho + warm_up, 1)
            return (x_tensor, c_tensor) if tuple_out else torch.cat((x_tensor, c_tensor), dim=2)

        return x_tensor

    def get_training_sample(
            self,
            dataset: Dict[str, np.ndarray],
            ngrid_train: int,
            nt: int
    ) -> Dict[str, torch.Tensor]:
        """Generate a training batch."""
        batch_size = self.config['train']['batch_size']
        i_sample, i_t = randomIndex_percentage(ngrid_train, (batch_size, self.rho),dataset['TTD'], warm_up=self.warm_up)

        return {
            'x_phy' : self.select_subset(dataset['x_phy'] , i_sample, i_t).to(self.device),
            'x_soil': self.select_subset(dataset['x_soil'], i_sample, i_t).to(self.device),
            'c_phy' : dataset['c_phy'][i_sample],
            'p_phy' : dataset['p_phy'][i_sample],
            'v_phy' : {key: value[i_sample] for key, value in dataset["mvars"].items()},
            'mconns': dataset['mconns'],

            'xc_nn_norm': self.select_subset(dataset['xc_nn_norm'], i_sample, i_t, has_grad=False).to(self.device),
            'target'    : self.select_subset(dataset['target'], i_sample, i_t)[self.warm_up:, :].to(self.device),
            'aux'       : self.model_sampler(dataset['aux'], i_sample),
            'paramsDict': dataset['paramsDict'],
            'dimsDict'  : dataset['dimsDict'],

            'batch_sample': i_sample,
            'pl_code'     : dataset['pl_code'][i_sample]
            # 'c_nn': dataset['c_nn'][i_sample],

        }

    def get_validation_sample(self, dataset: Dict[str, torch.Tensor], i_s: int, i_e: int) -> Dict[str, torch.Tensor]:
        """Generate a validation batch."""
        return {
            key: torch.tensor(
                value[self.warm_up:, i_s:i_e, :] if value.ndim == 3 else value[i_s:i_e, :],
                dtype=torch.float32,
                device=self.device
            )
            for key, value in dataset.items()
        }

    # def take_sample_old(self, dataset: Dict[str, torch.Tensor], days=730, basins=100) -> Dict[str, torch.Tensor]:
    #     """Take a sample for a specified time period and number of basins."""
    #     sample = {
    #         key: torch.tensor(
    #             value[self.warm_up:days, :basins, :] if value.ndim == 3 else value[:basins, :],
    #             dtype=torch.float32,
    #             device=self.device
    #         )
    #         for key, value in dataset.items()
    #     }
    #     # Adjust target for warm-up days if necessary
    #     if 'HBV1_1p' not in self.config['dpl_model']['phy_model']['model'] or not self.config['dpl_model']['phy_model']['warm_up_states']:
    #         sample['target'] = sample['target'][self.warm_up:days, :basins]
    #     return sample
    def model_sampler(self, modelsDict, iGrid):
        modelsDict = copy.deepcopy(modelsDict)

        for key in ['BCsMod', 'wrf_soil', 'wkf_soil', 'wrf_plant', 'wkf_plant']:
            module = modelsDict[key]
            if key == 'wkf_plant':
                for key1, val1 in module.items():
                    val1.updateAttrs(val1.subsetAttrs(iGrid))
                    modelsDict[key][key1] = val1
            else:
                module.updateAttrs(module.subsetAttrs(iGrid))
                modelsDict[key] = module

        return modelsDict


def take_sample(config: Dict, dataset_dict: Dict[str, torch.Tensor], days=730,
                basins=100) -> Dict[str, torch.Tensor]:
    """Take sample of data."""
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['dpl_model']['phy_model']['warm_up']
            dataset_sample[key] = torch.tensor(value[warm_up:days, :basins, :]).float().to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = torch.tensor(value[:basins, :]).float().to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    return dataset_sample

    # Keep 'warmup' days for dHBV1.1p.
    if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
            (config['dpl_model']['phy_model']['warm_up_states']) and (config['multimodel_type'] == 'none'):
        pass
    else:
        dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:days, :basins]
    return dataset_sample


# def take_sample_train(args, Input_dataDict, ngrid_train, batchSize, TTD_new):
#     dimSubset = [batchSize, args["rho"]]
#     iGrid, iT = randomIndex_percentage(ngrid_train, dimSubset, TTD_new, warm_up=args["warm_up"])
#     Input_dataDict  = copy.deepcopy(Input_dataDict)
#     sample_dataDict = dict()
#     sample_dataDict['iGrid'] = iGrid
#     sample_dataDict['pl_code'] = args['pl_code'][iGrid]
#     # Physical model inputs
#     sample_dataDict["x_PBM"]  = selectSubset(args, Input_dataDict["forcing"]['physical'], iGrid, iT, args["rho"], has_grad=False,warm_up=args["warm_up"], listIn=True)
#     sample_dataDict["x_soil"] = selectSubset(args, Input_dataDict["forcing"]['soil'], iGrid, iT, args["rho"], has_grad=False,warm_up=args["warm_up"], listIn=True)
#     sample_dataDict["c_PBM"]  = Input_dataDict["attrs"][iGrid]
#     sample_dataDict["p_PBM"]  = Input_dataDict["params"][iGrid]
#     sample_dataDict["v_PBM"]  = {key: value[iGrid] for key, value in Input_dataDict["mvars"].items()}
#     sample_dataDict['mconns'] = Input_dataDict['mconns']
#
#     # NN inputs
#     sample_dataDict["x_NN"]= selectSubset(args, Input_dataDict["x_NN"], iGrid, iT, args["rho"], has_grad=False,warm_up=args["warm_up"])
#     sample_dataDict["c_NN"]= selectSubset(args, Input_dataDict["c_NN"], iGrid, has_grad=False, iT= None, rho = None)
#     sample_dataDict["obs"] = selectSubset(args, Input_dataDict["obs"], iGrid, iT, args["rho"], has_grad=False,warm_up=args["warm_up"])
#     sample_dataDict["obs"] = normalize_sapflow(args, sample_dataDict["c_NN"], sample_dataDict["obs"] )
#     return sample_dataDict
#
# def take_sample_test(args, Input_dataDict, iGrid, iT):
#     Input_dataDict  = copy.deepcopy(Input_dataDict)
#     sample_dataDict = dict()
#     sample_dataDict['iGrid']  = [iGrid]
#     sample_dataDict['pl_code']= args['pl_code'][iGrid]
#
#     # Physical model inputs
#     sample_dataDict["x_PBM"]  = torch.tensor(Input_dataDict["forcing"]['physical'][iGrid][iT:],device=args["device"] ).unsqueeze(1)
#     sample_dataDict["x_soil"] = torch.tensor(Input_dataDict["forcing"]['soil'][iGrid][iT:],device=args["device"] ).unsqueeze(1)
#     sample_dataDict["c_PBM"]  = Input_dataDict["attrs"][iGrid:iGrid +1]
#     sample_dataDict["p_PBM"]  = Input_dataDict["params"][iGrid:iGrid +1]
#     sample_dataDict["v_PBM"]  = {key: value[iGrid: iGrid+1] for key, value in Input_dataDict["mvars"].items()}
#     sample_dataDict['mconns'] = Input_dataDict['mconns']
#
#     # NN inputs
#     sample_dataDict["x_NN"]   = torch.tensor(Input_dataDict["x_NN"][iGrid][iT:],device=args["device"] ).unsqueeze(1)
#     sample_dataDict["c_NN"]   = torch.tensor(Input_dataDict["c_NN"][iGrid: iGrid+1], device=args["device"])
#     sample_dataDict["obs"]    = torch.tensor(Input_dataDict["obs"][iGrid][iT:],device=args["device"] ).unsqueeze(1)
#
#     sample_dataDict["obs"]    = normalize_sapflow(args, sample_dataDict["c_NN"], sample_dataDict["obs"] )
#
#     return sample_dataDict
# def normalize_sapflow(args, c_NN_sample, obs_sample):
#     varTarget_NN   = args["target"]
#     norm_regime = args['norm_regime']
#     if "sapf" in varTarget_NN:
#         obs_flow_v = obs_sample[:, :, varTarget_NN.index("sapf")]
#         varC_NN = args["varC_NN"]
#         if norm_regime == 0: # normalize by sapwarea ----> cm/hr
#             factor_name = "pl_sapw_area"
#             factor = c_NN_sample[:, varC_NN.index(factor_name)]
#             factor = factor * 10**4 # convert to cm2
#         elif norm_regime ==1: # normalize by pl_dbh ----> cm2/hr
#             factor_name = "pl_dbh"
#             factor = c_NN_sample[:, varC_NN.index(factor_name)]
#         elif norm_regime ==2: # normalize by pl_dbh and sapw_area ----> 1/hr
#             pl_dbh    = c_NN_sample[:, varC_NN.index("pl_dbh")]
#             sapw_area = c_NN_sample[:, varC_NN.index("pl_sapw_area")]
#             factor    = pl_dbh * sapw_area * 10**4
#         else:
#             raise ValueError("please chooose proper norm regime method")
#         obs_sample[:, :, varTarget_NN.index("sapf")] = obs_flow_v / factor.unsqueeze(dim=0) # convert cm3/hr to cm2/hr or cm/hr
#     return obs_sample
