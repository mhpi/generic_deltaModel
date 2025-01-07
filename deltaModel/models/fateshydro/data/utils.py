from conf.Constants import fates_r8, fates_int
import os
import torch
import numpy as np
import pandas as pd
import datetime
import json
def createTensor(dims, dtype, fill_nan = False, requires_grad=False, value = None):
    """
    A small function to centrally manage device, data types, etc., of new arrays
    """
    if value is None:
        tensor = torch.zeros(dims,requires_grad=requires_grad,dtype=dtype)
    else:
        tensor =  torch.zeros(dims, requires_grad=requires_grad, dtype=dtype) + value

    if fill_nan:
        tensor.fill_(torch.nan)

    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def dissolve_value(value, nlayers):
    # Check if the value exceeds the allowed maximum
    if value > nlayers:
        print(value)
        raise ValueError("Value must be less than the number of layers")

    # Initialize parts array with zeros, size based on nlayers
    parts = [0] * nlayers

    # Distribute the value across the parts array
    for i in range(np.int64(np.ceil(value))):
        if value>1:
            parts[i] = 1
        elif value >0:
            parts[i] = value
        else:
            parts[i] = 0
        value -= 1

    # print(value, parts)
    return parts


def extract_lai_attrs(lai_in):
    lai_in     = torch.where(torch.isnan(lai_in), 0, lai_in)
    nlayers    = np.int64(np.ceil(torch.max(lai_in).item()))

    vectorized_LAI = np.vectorize(dissolve_value, otypes=[np.ndarray])
    lai_out        = vectorized_LAI(lai_in.to("cpu").numpy(), nlayers)
    lai_out        = torch.tensor(np.array(lai_out.tolist()).reshape(*lai_in.shape, nlayers)).to(lai_in.device)

    cumulative_lai      = torch.zeros_like(lai_out)
    cumulative_lai_node = torch.zeros_like(lai_out)

    # GET the cummulative leaf area index above each leaf layer
    for c_ly in range(nlayers):
        lai_lsl = lai_out[:, :, c_ly]
        if c_ly == 0:
            # zero layers above the first layer
            cumulative_lai[:, :, c_ly] = 0
        else:
            cumulative_lai[:, :, c_ly] = torch.where(lai_lsl > 0,torch.sum(lai_out[:, :, :c_ly], dim=-1),1.e36)  # added very large number to ensurenzero radiation at non exisiting layers

        # GET the cummulative leaf area index at the intermediate of the leaf layer
        cumulative_lai_node[:, :, c_ly] = cumulative_lai[:, :, c_ly] + lai_lsl / 2.0

    appended       = lai_out.sum(dim=-1).unsqueeze(dim=-1)
    cumulative_lai = torch.cat((cumulative_lai, appended), dim=-1)
    cumulative_lai = torch.where(cumulative_lai == 1.e36, appended, cumulative_lai)

    return lai_out, cumulative_lai, cumulative_lai_node

########################################################################################################################
########################################################################################################################
####################################################DATA NORMALIZATION##################################################
########################################################################################################################
########################################################################################################################
def scale_params(x, p_ranges):
    x_out = x.clone()
    for p in range(x_out.shape[-1]):
        x_out[...,p] = x[...,p] * (p_ranges['max'][p] - p_ranges['min'][p]) + p_ranges['min'][p]

    return x_out


def percentage_day_cal(ttd):
    trees = ttd['pl_code'].values
    tempData = []

    for ind, pl in enumerate(trees):
        S_training = ttd.loc[ttd['pl_code'] == pl, 'S_Training']
        E_training = ttd.loc[ttd['pl_code'] == pl, 'E_Training']
        d1 = datetime.datetime(S_training[ind].year, S_training[ind].month, S_training[ind].day, S_training[ind].hour, S_training[ind].minute)  
        d2 = datetime.datetime(E_training[ind].year, E_training[ind].month, E_training[ind].day, E_training[ind].hour, E_training[ind].minute)
        delta = d2 - d1
        hours_delta = delta.total_seconds() / 3600
        tempData.append(hours_delta)

    temp = pd.Series(tempData)
    ttd['no_hours'] = temp
    total_hours = np.sum(temp)
    tempPercent = []
    for pl in trees:
        hours = ttd.loc[ttd['pl_code'] == pl, 'no_hours'].values[0]
        tempPercent.append(hours/total_hours)
    temp1 = pd.Series(tempPercent)
    ttd['hour_percent'] = temp1
    return ttd

