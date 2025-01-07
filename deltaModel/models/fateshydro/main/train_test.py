import os
import copy
import time
import torch
import pickle
import pandas as pd
import numpy as np
from post import Stat, plot
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.datahandler import No_iter_nt_ngrid, take_sample_train, take_sample_test, normalize_sapflow
from data.utils       import init_norm_stats, transNorm_tensor, scale_params

from models.physical.Fateshydro import photo_Hydro_Module
from models.NN.set_loss_func    import get_lossFun


def train_Model(args, datasetDict, modelsDict):
    NN_model = modelsDict['NN_model']
    optim    = modelsDict['optim']

    lossFun  = get_lossFun(args)  # obs is needed for certain loss functions, not all of them

    ngrid_train, nIterEp, nt, batchSize, TTD_new = No_iter_nt_ngrid(datasetDict['TTD'], args, datasetDict['attrs'])

    ### normalizing
    # creating the stats for normalization of NN inputs
    init_norm_stats(args, datasetDict["x_NN"], datasetDict["c_NN"], datasetDict["obs"], TTD_new)

    if torch.cuda.is_available():
        NN_model = modelsDict['NN_model'].to(args["device"])
        lossFun  = lossFun.to(args["device"])
        torch.backends.cudnn.deterministic = True

    NN_model.zero_grad()
    NN_model.train()
    # training
    for epoch in range(1, args["EPOCHS"] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            sample_dataDict    = take_sample_train(args, datasetDict, ngrid_train, batchSize, TTD_new)
            sample_modelsDict  = extract_sample_model(modelsDict, sample_dataDict['iGrid'])
            # normalize
            x_NN_scaled = transNorm_tensor(args, sample_dataDict["x_NN"], varLst=args["varT_NN"], toNorm=True)
            c_NN_scaled = transNorm_tensor(args, sample_dataDict["c_NN"], varLst=args["varC_NN"], toNorm=True)
            c_NN_scaled = c_NN_scaled.unsqueeze(0).repeat(x_NN_scaled.shape[0], 1, 1)
            sample_dataDict["inputs_NN_scaled"] = torch.concatenate((x_NN_scaled, c_NN_scaled), dim=2).float() # convert to float32
            obs_scaled  = sample_dataDict['obs']

            # Batch running of the differentiable model
            params        = NN_model(sample_dataDict)
            params_scaled = scale_params(params, args['out_NN_ranges'])# params + 0.001 #
            physical_model= photo_Hydro_Module(args)
            physical_model(args, sample_dataDict, sample_modelsDict, params_scaled)

            # loss function
            sim        = physical_model.forward_nsteps(args['dtime'],sample_modelsDict, params_scaled) *1000 # convert from kg/hr to cm3/hr
            sim_scaled = normalize_sapflow(args, sample_dataDict["c_NN"], sim)
            # sim_scaled = transNorm_tensor(args, sim_scaled, varLst=args["target"], toNorm=True)
            loss = lossFun(sim_scaled[args['warm_up']:],obs_scaled[args['warm_up']:])
            loss.backward()
            optim.step()
            NN_model.zero_grad()
            lossEp = lossEp + loss.item()
            if (iIter % 1 == 0) or (iIter == nIterEp):
                print(iIter, " from ", nIterEp, " in the ", epoch,"th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(epoch, lossEp, time.time() - t0,int(torch.cuda.memory_allocated(device=args["device"]) * 0.001))
        print(logStr)

        if epoch % args["saveEpoch"] == 0:
            # save model
            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
            torch.save(NN_model, modelFile)
        if epoch == args["EPOCHS"]:
            print("last epoch")
    print("Training ended")
    return

def test_Model(args, datasetDict, modelsDict):
    ngrid_train, nIterEp, nt, batchSize, TTD_new = No_iter_nt_ngrid(datasetDict['TTD'], args, datasetDict['attrs'])
    modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCH_testing"]) + ".pt")
    NN_model  = torch.load(modelFile)
    NN_model.eval()
    physical_model = photo_Hydro_Module(args)
    ngrid = len(datasetDict["x_NN"])

    list_sim = []
    list_obs = []
    for ii in range(0, ngrid): #[3,6,26,51,88]:
        print(ii)
        # nt_site = 0
        nt_site = int(TTD_new.iloc[ii]['no_hours']) # - args['warm_up']

        sample_dataDict   = take_sample_test(args, datasetDict, ii, nt_site)
        sample_modelsDict = extract_sample_model(modelsDict, sample_dataDict['iGrid'])
        # normalize
        x_NN_scaled = transNorm_tensor(args, sample_dataDict["x_NN"], varLst=args["varT_NN"], toNorm=True)
        c_NN_scaled = transNorm_tensor(args, sample_dataDict["c_NN"], varLst=args["varC_NN"], toNorm=True)
        c_NN_scaled = c_NN_scaled.unsqueeze(0).repeat(x_NN_scaled.shape[0], 1, 1)
        sample_dataDict["inputs_NN_scaled"] = torch.concatenate((x_NN_scaled, c_NN_scaled),dim=2).float()  # convert to float32

        # Batch running of the differentiable model
        out_NN = NN_model(sample_dataDict)
        params_scaled = scale_params(out_NN, args['out_NN_ranges']) #out_NN + 0.001#
        physical_model(args, sample_dataDict, sample_modelsDict, params_scaled)

        # loss function
        sim        = torch.stack(physical_model.out['sapflow'], dim = 0) * 1000  # convert from kg/hr to cm3/hr
        sim_scaled = normalize_sapflow(args, sample_dataDict["c_NN"], sim )  # convert from kg/hr to cm3/hr
        list_sim.append(sim_scaled.detach().to("cpu").numpy()[args['warm_up']:])
        list_obs.append(sample_dataDict['obs'].detach().to("cpu").numpy()[args['warm_up']:])
    save_outputs_Lsts(args, list_sim, list_obs, calculate_metrics=True)
    return

def forward_model_single(args, datasetDict, modelsDict):

    physical_model = photo_Hydro_Module(args)
    ngrid = len(datasetDict["x_NN"])
    nt_site = 0

    for ii in range(0, ngrid):
        print(ii)

        sample_dataDict   = take_sample_test(args, datasetDict, ii, nt_site)
        sample_modelsDict = extract_sample_model(modelsDict, sample_dataDict['iGrid'])
        # normalize
        x_NN_scaled = transNorm_tensor(args, sample_dataDict["x_NN"], varLst=args["varT_NN"], toNorm=True)
        c_NN_scaled = transNorm_tensor(args, sample_dataDict["c_NN"], varLst=args["varC_NN"], toNorm=True)
        c_NN_scaled = c_NN_scaled.unsqueeze(0).repeat(x_NN_scaled.shape[0], 1, 1)
        sample_dataDict["inputs_NN_scaled"] = torch.concatenate((x_NN_scaled, c_NN_scaled),dim=2).float()  # convert to float32
        syn_tensor = torch.tensor([0.0022, 0.8634, 0.9039, 0.8409, 0.3358, 0.7613, 0.0315, 0.3763, 0.3983,
                                   0.3234, 0.7100, 0.2722, 0.4420, 0.3556, 0.0305, 0.4174, 0.0763, 0.7916,
                                   0.7672, 0.9684]).to(x_NN_scaled)
        param        = torch.sum(sample_dataDict['inputs_NN_scaled'] * syn_tensor.unsqueeze(0).unsqueeze(0), dim = -1) #torch.sum(sample_dataDict["inputs_NN_scaled"]**2, dim = -1)
        param_scaled = torch.sigmoid(param) + 0.001
        physical_model(args, sample_dataDict, sample_modelsDict, param_scaled.unsqueeze(dim=-1))

        # loss function
        sim        = torch.stack(physical_model.out['sapflow'], dim = 0) * 1000  # convert from kg/hr to cm3/hr
        df_sapflow = pd.DataFrame(sim.squeeze(dim=1).to('cpu').detach().numpy())
        df_sapflow.columns = [sample_dataDict['pl_code']]
        df_sapflow.to_csv(os.path.join(args['out_path_syn'], f"{sample_dataDict['pl_code']}_sapf_H_syn.csv"), index=False)

    return

def forward_Model_batch(args, datasetDict, modelsDict):
    ngrid_train, nIterEp, nt, batchSize, TTD_new = No_iter_nt_ngrid(datasetDict['TTD'], args, datasetDict['attrs'])
    sample_dataDict       = take_sample_train(args, datasetDict, ngrid_train, batchSize, TTD_new)
    sample_modelsDict     = extract_sample_model(modelsDict, sample_dataDict['iGrid'])
    physical_model        = photo_Hydro_Module(args)

    start = time.time()
    physical_model(args, sample_dataDict, sample_modelsDict)
    sapf_sim = physical_model.out['sapflow']
    end = time.time()
    print(end-start)

    df_sapflow = pd.DataFrame(torch.stack(sapf_sim, dim=-1).squeeze(dim=1).to('cpu').detach().numpy()).transpose() #*1000
    df_sapflow.columns = list(sample_dataDict['pl_code'])
    df_sapflow.to_csv(f"batch{args['batch_size']}_sapf_v1.3_sim.csv", index=False)

    # df_sapflow = pd.DataFrame(sample_dataDict['obs'].squeeze(dim=-1).to( 'cpu').detach().numpy())
    # df_sapflow.columns = list(sample_dataDict['pl_code'])
    # df_sapflow.to_csv(f"batch{args['batch_size']}_sapf_v1.3_obs.csv", index=False)
    return

def save_outputs_Lsts(args, predLst, obsLst, calculate_metrics=True):

    file_name = "NN_sim" + ".npy"
    with open(os.path.join(args["out_dir"], file_name), 'wb') as f: #, args["testing_dir"]
        pickle.dump(predLst, f)

    file_name = args['target'][0] + ".npy"
    with open(os.path.join(args["out_dir"], file_name), 'wb') as f: #, args["testing_dir"]
        pickle.dump(obsLst, f)

    if calculate_metrics == True:
        name_list = []

        if "sapf" in args["target"]:
            name_list.append("sapf")


        statDictLst = [
            stat.statError_Lsts(predLst, obsLst)
        ]
        ### save this file
        # median and STD calculation
        for st, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(st), 3])
            for key in st.keys():
                median = np.nanmedian(st[key])  # abs(i)
                STD = np.nanstd(st[key])  # abs(i)
                mean = np.nanmean(st[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=st.keys(), columns=["median", "STD", "mean"]
            )
            mdstd.to_csv((os.path.join(args["out_dir"], "mdstd_" + name + ".csv"))) #, args["testing_dir"]

            # Show boxplots of the results
            plt.rcParams["font.size"] = 14
            # keyLst = ["Bias", "RMSE", "ubRMSE", "NSE", "Corr"]
            keyLst = ["Bias", "RMSE", "KGE", "NSE", "Corr"]

            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                # for k in range(len(st)):
                data = st[statStr]
                data = data[~np.isnan(data)]
                temp.append(data)
                dataBox.append(temp)
            labelname = [
                "LSTM model"
            ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            xlabel = ["Bias (cm3/hr)", "RMSE", "KGE", "NSE", "Corr"]
            fig = plot.plotBoxFig(
                dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
            )
            fig.patch.set_facecolor("white")
            boxPlotName = "Sapflow"
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams["font.size"] = 12
            plt.savefig(
                os.path.join(args["out_dir"], "Box_" + name + ".png") #, args["testing_dir"]
            )  # , dpi=500
            # fig.show()
            plt.close()

def extract_sample_model(modelsDict, iGrid):
    modelsDict = copy.deepcopy(modelsDict)

    for key in ['BCsMod', 'wrf_soil','wkf_soil', 'wrf_plant', 'wkf_plant']:
        module = modelsDict[key]
        if key == 'wkf_plant':
            for key1,val1 in module.items():
                val1.updateAttrs(val1.subsetAttrs(iGrid))
                modelsDict[key][key1] = val1
        else:
            module.updateAttrs(module.subsetAttrs(iGrid))
            modelsDict[key] = module


    return modelsDict


