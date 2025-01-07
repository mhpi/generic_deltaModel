import os
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from conf.Constants     import *
from data.utils         import extract_lai_attrs, createTensor, percentage_day_cal
from concurrent.futures import ThreadPoolExecutor
from physical.biogeophys.Functionals import getnscaler


class data_reader():
    def __init__(self,args):
        attr_path    = args['observations']['attr_path']
        forcing_path = args['observations']['forcing_path']
        T_T_path     = args['observations']['T_T_path']
        attrLst      = args['dpl_model']['nn_model']['attributes']
        x_nn_Lst     = args['dpl_model']['nn_model']['forcings']
        x_phy_Lst    = args['dpl_model']['phy_model']['forcings']
        target_Lst   = args['train']['target']

        varLists = {
            'forcing': x_phy_Lst,
            'target' : target_Lst,
            'x_nn'   : x_nn_Lst
        }

        attrs  = self.InitAttrs(attr_path)
        c_nn   = self.readAttrs(attr_path, varLst= attrLst)
        TTD    = self.readTTD(T_T_path, attrs)
        paramsDict = self.readParamsDict()
        dimsDict   = self.InitdimsDict(args)
        params     = self.InitParams(paramsDict, attrs)
        forcing, target, x_nn = self.readTs(attrs['pl_code'], varLists, forcing_path)

        self.dataDict = {
            'attrs'  : attrs,
            'params' : params,
            'forcing': forcing,
            'target' : target,
            'c_nn'   : c_nn ,
            'x_nn'   : x_nn  ,
            'TTD'    : TTD,
            'paramsDict' : paramsDict,
            'dimsDict'   : dimsDict
        }
        return

    def readTTD(self, path, attrs):
        ttd = pd.read_excel(path)#.iloc[:100] #.iloc[604:636]#.iloc[712:722]
        ttd = ttd[ttd['pl_code'].isin(attrs['pl_code'])]
        ttd = ttd.reset_index(drop = True)
        ttd = percentage_day_cal(ttd)

        return ttd
    def readAttrs(self, path, varLst = None):
        # read attrs from each tree (ntrees X nattrs)
        attrs = pd.read_csv(path)#.iloc[:100] #.iloc[604:636]#
        attrs = attrs[attrs['pl_lai'] <=5]
        if varLst is not None:
            attrs = attrs[varLst].values
        return attrs

    def readTs(self, trees,varLst, dirDB):
        with ThreadPoolExecutor() as executor:
            # Wrap the iterable with tqdm to show progress
            results = list(tqdm(
                executor.map(lambda tree: self.readTsSite(tree, varLst, dirDB), trees),
                total=len(trees),  # Total number of tasks
                desc="Reading Data",  # Description for the progress bar
                unit="tree"  # Unit of measurement
            ))
        forcing = [res['forcing'] for res in results]
        target  = [res['target'] for res in results]
        x_nn    = [res['x_nn'] for res in results]

        return forcing, target, x_nn

    def readTsSite(self,tree, varLst, dirDB):
        dataFile = os.path.join(dirDB, f"{tree}_forcing_H.csv")
        allvars = set(varLst['forcing'] + varLst['target'] + varLst['x_nn'])
        dataTemp = pd.read_csv(dataFile,  usecols=list(allvars))
        out = {
            'forcing': dataTemp[varLst['forcing']].values,
            'target' : dataTemp[varLst['target']].values,
            'x_nn'   : dataTemp[varLst['x_nn']].values
        }
        return out

    def InitAttrs(self, path):

        attrs = self.readAttrs(path)
        attrsDict = {}

        " Reading plant attributes"
        for key in ['pl_pft', 'pl_dbh','pl_leaf_area', 'pl_sapw_area','pl_height', 'pl_lai','pl_crown_area','st_density'  ]:
            dtype = fates_int if key == 'pl_pft' else fates_r8
            attrsDict[key] = torch.tensor(attrs[key].values,dtype=dtype).cuda().unsqueeze(dim=-1)

        " Reading soil attributes"
        for key in ['clay', 'sand', 'soc']:
            attrsDict[key] = torch.tensor(attrs[[f'{key}_0-10cm_mean', f'{key}_10-30cm_mean', f'{key}_30-60cm_mean',f'{key}_60-100cm_mean', f'{key}_100-200cm_mean']].values, dtype=fates_r8).cuda()
            if key == "soc":
                attrsDict['om_frac'] =  attrsDict.pop('soc')
                attrsDict['om_frac'] = attrsDict['om_frac']/ 2 / 100 * 1.72

        " Reading plant and site codes"
        attrsDict['pl_code']      = attrs['pl_code'].values
        attrsDict['si_code']      = attrs['si_code'].values

        " Reading soil discretization"
        for key, value in zip(['dz', 'zi', 'zsoi'], [dz,zi,zsoi]):
            attrsDict[key]   = torch.tensor([value], dtype=fates_r8).cuda().expand(attrs.shape[0],-1)
            # zi : Define the depth to each soil layer, originally in Fates it has zero index with zero value but neglected here
            # xsoi: Define the depth to the mid of each soil layer

        attrsDict['lai'], attrsDict['cummulative_lai'], attrsDict['cummulative_lai_node'] = extract_lai_attrs(attrsDict['pl_lai'])
        return attrsDict

    def InitParams(self, prtparamsDict, attrs):
        dims     = attrs['pl_pft'].shape
        pftnames = prtparamsDict['pftname']

        params = {param: createTensor(dims=dims, dtype=fates_r8, value=fates_unset_r8).cuda() for param in prtparamsDict['pftparams'].keys()}
        for ft_id in range(len(pftnames)):
            mask = (attrs['pl_pft'] == ft_id).bool()

            for key,val in prtparamsDict['pftparams'].items():
                params[key] = torch.where(mask, val[ft_id], params[key])

        params['nscaler'] = getnscaler(params['vcmax25top_ft'].unsqueeze(dim=-1),
                                       params['scaler_coeff1'].unsqueeze(dim=-1),
                                       params['scaler_coeff2'].unsqueeze(dim=-1),
                                       attrs['cummulative_lai_node'])

        return params

    def readParamsDict(self):
        with open(Path(__file__).parent.parent / "conf/params.json") as json_file:
            Dict = json.load(json_file)

        return Dict

    def InitdimsDict(self, args):
        with open(Path(__file__).parent.parent / "conf/dimsDict.json") as json_file:
            dimsDict = json.load(json_file)

        aggmeth = args['dpl_model']['phy_model']['aggmeth']

        nlevsoil = dimsDict['nlevsoil']
        n_hypool_leaf = dimsDict['n_hypool_leaf']
        n_hypool_stem = dimsDict['n_hypool_stem']
        n_hypool_aroot = dimsDict['n_hypool_aroot']
        n_hypool_troot = dimsDict['n_hypool_troot']
        nshell = dimsDict['nshell']

        # Number of aboveground plant water storage nodes
        n_hypool_ag = n_hypool_leaf + n_hypool_stem

        # Get nlevrhiz
        if aggmeth == "rhizlayer_aggmeth_none":
            nlevrhiz = nlevsoil
        elif aggmeth == "rhizlayer_aggmeth_combine12":
            nlevrhiz = max(1, nlevsoil - 1)
        elif aggmeth == "rhizlayer_aggmeth_balN":
            nlevrhiz = min(dimsDict['aggN'], nlevsoil)
        else:
            raise RuntimeError('You specified an undefined rhizosphere layer aggregation method')

        dimsDict['num_connections'] = n_hypool_leaf + n_hypool_stem + n_hypool_troot - 1 + (
                    n_hypool_aroot + nshell) * nlevrhiz
        dimsDict['num_nodes'] = n_hypool_leaf + n_hypool_stem + n_hypool_troot + (n_hypool_aroot + nshell) * nlevrhiz
        dimsDict['n_hypool_plant'] = n_hypool_leaf + n_hypool_stem + n_hypool_troot + (n_hypool_aroot * nlevrhiz)
        dimsDict['n_hypool_ag'] = n_hypool_ag
        dimsDict['nlevrhiz'] = nlevrhiz


        return dimsDict
