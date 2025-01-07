import copy
import json
import logging
import os
import pickle
from math import e, log
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.exceptions import DataDimensionalityWarning

from core.data import intersect
from core.data.data_loaders.base import BaseDataLoader
from models.fateshydro.Init.InitMod import InitMod

log = logging.getLogger(__name__)


class fatesDataLoader(BaseDataLoader):
    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite  = overwrite
        self.target        = config['train']['target']
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings   = config['dpl_model']['nn_model'].get('forcings', [])
        self.log_norm_vars = config['dpl_model']['phy_model']['use_log_norm']
        self.norm_regime   = config['dpl_model']['phy_model']['norm_regime']

        self.device = config['device']
        self.dtype  = config['dtype']
        self.train_dataset = None
        self.eval_dataset  = None
        self.dataset       = None
        self.norm_stats    = None
        self.out_path      = os.path.join(
            config['out_path'],
            'normalization_statistics.json',
        )
        self.load_dataset()

        return

    def _split_data(self,data, scope):
        if self.test_split:
            ttd = data['TTD']

            if scope == 'train':
                for key in ['x_nn', 'x_phy', 'x_soil', 'target']:
                    temp = data[key].copy()
                    val  = [array[:int(idx)] for array, idx in zip(temp, ttd['no_hours'].values)]
                    data[key] = val
            elif scope == 'test':
                for key in ['x_nn', 'x_phy', 'x_soil', 'target']:
                    temp = data[key].copy()
                    val  = [array[int(idx):] for array, idx in zip(temp, ttd['no_hours'].values)]
                    data[key] = val
            else:
                raise ValueError('scope definition is unknown')
        else:
            pass
        return data

    def load_dataset(self) -> None:
        """Load dataset into dictionary of nn and physics model input arrays."""
        if self.test_split:
            tempDict = self._preprocess_data(scope='traintest')
            self.train_dataset, self.eval_dataset= tempDict[0], tempDict[1]
            del tempDict
        else:
            self.dataset = self._preprocess_data(scope='all')[0]

    def _preprocess_data(self, scope: Optional[str]) -> Tuple[npt.NDArray]:
        """Read data from the dataset."""
        #x_phy, c_phy, x_nn, c_nn, target = self.read_data(scope)
        initializer = InitMod(self.config)
        datasetsDict= []

        if self.test_split:
            scope_list = ['train', 'test']
        else:
            scope_list = ['all']

        for scope in scope_list:
            data = self._split_data(initializer.data, scope)
            aux  = initializer.models_Dict
            x_nn, c_nn, target = data['x_nn'], data['c_nn'], data['target']
            target = self._flow_conversion(c_nn, target)

            # Normalize nn input data
            self.load_norm_stats(x_nn, c_nn, target)
            xc_nn_norm = self.normalize(x_nn, c_nn)

            # Build data dict of Torch tensors
            dataset = copy.deepcopy(data)
            del dataset['x_nn']
            del dataset['c_nn']
            del dataset['target']
            # dataset['x_phy']        = [arr for arr in dataset['x_phy']],#[arr.to(self.device) for arr in dataset['x_phy']]
            # dataset['x_soil']       = [arr for arr in dataset['x_soil']],#[arr.to(self.device) for arr in dataset['x_soil']]
            # dataset['target']       = [self.to_tensor(arr) for arr in target]
            # dataset['xc_nn_norm']   = [self.to_tensor(arr) for arr in xc_nn_norm]
            dataset['target']       = [torch.from_numpy(arr).to(dtype=self.dtype) for arr in target]
            dataset['xc_nn_norm']   = [torch.from_numpy(arr).to(dtype=self.dtype) for arr in xc_nn_norm]
            dataset['aux']          = aux

            datasetsDict.append(dataset)
        del initializer
        del data
        del aux

        return datasetsDict

    def load_norm_stats(
            self,
            x_nn: List[npt.NDArray],
            c_nn: npt.NDArray,
            target: List[npt.NDArray],
    ) -> None:
        """Load or calculate normalization statistics if necessary."""
        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                with open(self.out_path, 'r') as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)

    def _init_norm_stats(
            self,
            x_nn: List[npt.NDArray],
            c_nn: npt.NDArray,
            target: List[npt.NDArray],
    ) -> Dict[str, List[float]]:
        """Compile calculations of data normalization statistics."""
        stat_dict = {}

        # Get basin areas from attributes.
        # basin_area = self._get_basin_area(c_nn)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            x_toNorm = [array[:, k] for array in x_nn]
            x_toNorm = np.concatenate(x_toNorm)
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_toNorm)
            else:
                stat_dict[var] = self._calc_norm_stats(x_toNorm)

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])

        # Target variable stats
        for i, name in enumerate(self.target):
            y_toNorm = [array[:, i] for array in target]
            y_toNorm = np.concatenate(y_toNorm)
            if name in self.log_norm_vars:
                stat_dict[name] = self._calc_gamma_stats(y_toNorm)
            else:
                stat_dict[name] = self._calc_norm_stats(y_toNorm)
            # if name in ['sapf_sim', 'sapf']:
            #     stat_dict[name] = self._calc_norm_stats(
            #         np.swapaxes(target[:, :, i:i + 1], 1, 0).copy(),
            #         basin_area,
            #     )
            # else:
            #     stat_dict[name] = self._calc_norm_stats(
            #         np.swapaxes(target[:, :, i:i + 1], 1, 0),
            #     )

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)

        return stat_dict

    def _calc_norm_stats(
            self,
            x: npt.NDArray,
    ) -> List[float]:
        """
        Calculate statistics for normalization with optional basin
        area adjustment.
        """
        # Handle invalid values
        x[x == -999] = np.nan

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            b = np.array([0])

        # Calculate statistics
        transformed = b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x: npt.NDArray) -> List[float]:
        """Calculate gamma statistics for streamflow and precipitation data."""
        a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
        b = a[(~np.isnan(a))]
        b[b<0] = 0
        b = np.log10(
            np.sqrt(b) + 0.1
        )

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def normalize(self, x_nn: Union[npt.NDArray, List], c_nn: npt.NDArray) -> Union[npt.NDArray, List]:
        """Normalize data for neural network."""
        x_nn_norm = self._to_norm(
            x_nn.copy(),
            self.nn_forcings,
        )
        c_nn_norm = self._to_norm(
            c_nn,
            self.nn_attributes,
        )

        if isinstance(x_nn, list):
            xc_nn_norm = []
            for i in range(len(x_nn_norm)):
                x_nn_temp = x_nn_norm[i]
                c_nn_temp = c_nn_norm[i:i+1, :]
                # Remove nans
                x_nn_temp[x_nn_temp != x_nn_temp] = 0
                c_nn_temp[c_nn_temp != c_nn_temp] = 0

                c_nn_temp = np.repeat(
                    #np.expand_dims(c_nn_temp, 0),
                    c_nn_temp,
                    x_nn_temp.shape[0],
                    axis=0
                )

                xc_nn_norm.append(np.concatenate((x_nn_temp, c_nn_temp), axis=-1))

        else:
            # Remove nans
            x_nn_norm[x_nn_norm != x_nn_norm] = 0
            c_nn_norm[c_nn_norm != c_nn_norm] = 0

            c_nn_norm = np.repeat(
                np.expand_dims(c_nn_norm, 0),
                x_nn_norm.shape[0],
                axis=0
            )

            xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
        del x_nn_norm, c_nn_norm, x_nn

        return xc_nn_norm

    def _to_norm(self, data: Union[npt.NDArray, List], vars: List[str]) -> Union[npt.NDArray, List]:
        """Standard data normalization."""
        # Copy the input array to avoid modifying the original
        x_temp = data.copy()

        # Pre-compute statistics for all variables
        means = np.array([self.norm_stats[var][2] for var in vars])#, dtype=data.dtype)
        stds  = np.array([self.norm_stats[var][3] for var in vars])#, dtype=data.dtype)

        # Apply transformations based on toNorm
        # Identify variables requiring log transformation (precip, sapf)
        mask_log_norm = [var in self.log_norm_vars for var in vars]
        mask_log_norm = np.array(mask_log_norm)  # Convert to array

        if isinstance(x_temp, list):
            out = []
            for arr in x_temp:
                # Apply log transformation for specified variables
                if mask_log_norm.any():
                    arr[..., mask_log_norm] = np.log10(np.sqrt(arr[..., mask_log_norm]) + 0.1)
                out.append((arr - means)/stds)
        else:

            # Apply log transformation for specified variables
            if mask_log_norm.any():
                x_temp[..., mask_log_norm] = np.log10(np.sqrt(x_temp[..., mask_log_norm]) + 0.1)

            # Normalize
            out = (x_temp - means) / stds

        return out

    def _from_norm(self, data: Union[npt.NDArray, List], vars: List[str]) -> Union[npt.NDArray, List]:
        """Standard data normalization."""
        # Copy the input array to avoid modifying the original
        x_temp = np.copy(data)

        # Pre-compute statistics for all variables
        means = np.array([self.norm_stats[var][2] for var in vars], dtype=data.dtype)
        stds = np.array([self.norm_stats[var][3] for var in vars], dtype=data.dtype)

        # Apply transformations based on toNorm
        # Identify variables requiring log transformation (precip, sapf)
        mask_log_norm = [var in self.log_norm_vars for var in vars]
        mask_log_norm = np.array(mask_log_norm)  # Convert to array

        if isinstance(x_temp, list):
            out = []
            for arr in out:
                temp = arr * stds + means
                # Revert log transformation for specified variables
                if mask_log_norm.any():
                    temp[..., mask_log_norm] = (np.pow(10, temp[..., mask_log_norm]) - 0.1) ** 2
                out.append(temp)
        else:
            # Revert normalization
            out = x_temp * stds + means

            # Revert log transformation on specific variables
            mask_log_norm = [var in self.log_norm_vars for var in vars]
            out[..., mask_log_norm] = (np.pow(10, out[..., mask_log_norm]) - 0.1) ** 2

        return out

    def _flow_conversion(self, c_nn, target) -> List[npt.NDArray]:
        target_out = target.copy()
        for name in ['sapf', 'sapf_sim']:
            if name in self.target:
                for t in range(len(target)):
                    target_temp = target[t][:, self.target.index(name)]
                    if self.norm_regime == 0:
                        factor_name = "pl_sapw_area"
                        factor      = c_nn[t, self.nn_attributes.index(factor_name)]
                        factor      = factor * 10 ** 4  # convert from m2 to cm2
                    elif self.norm_regime == 1:
                        factor_name = "pl_dbh"
                        factor      =  c_nn[t, self.nn_attributes.index(factor_name)]
                    elif self.norm_regime == 2:
                        pl_dbh       = c_nn[t, self.nn_attributes.index("pl_dbh")]
                        pl_sapw_area = c_nn[t, self.nn_attributes.index("pl_sapw_area")]
                        factor = pl_dbh * pl_sapw_area * 10 ** 4  # convert from m2 to cm2
                    else:
                        raise ValueError("please choose proper norm regime method")
                    
                    target_temp = target_temp/factor
                    target_out[t][:, self.target.index(name)] = target_temp
        return target_out
    
            




