import copy
import math
import logging

from conf.Constants import *
from data.data_reader import data_reader
from Init import InitFuncs as F_init
from physical.biogeophys import Functionals as F
from physical.soil.BCsMod import bc_in_type

#=======================================================================================================================
"""
This a wrapper module that initiates model variables, states, and modules
"""
#=======================================================================================================================
# A class to represent the site hydraulics
class InitMod():

    def __init__(self, args):
        data             = data_reader(args).dataDict
        dimsDict         = data['dimsDict']
        paramsDict       = data['paramsDict']

        self.mvars       = {}
        self.mstates     = {}
        self.models_Dict = {}

        self.InitBCsMod(args, data['attrs'])
        self.InitVars(args  , dimsDict, paramsDict, data['attrs'], data['params'], self.models_Dict['BCsMod'])
        self.InitWTFs(args  , dimsDict, paramsDict, data['attrs'], self.models_Dict['BCsMod'])

        data['dimsDict']['nlayers'] = data['attrs']['lai'].shape[-1]
        forcing = self.get_forcing(args, data['forcing'], data['params'], data['attrs'])

        self.data = {
            'c_phy'  : self.get_attrs(data['attrs']),
            'p_phy'  : self.get_params(data['params']),
            'mvars'  : self.get_mvars(),
            'mconns' : self.get_mconn(),
            'x_phy'  : forcing['physical'],
            'x_soil' : forcing['soil'],
            'target' : data['target'],
            'c_nn'   : data['c_nn'],
            'x_nn'   : data['x_nn'],
            'TTD'    : data['TTD'],
            'paramsDict' : data['paramsDict'],
            'dimsDict'   : data['dimsDict'],
            'pl_code' : data['attrs']['pl_code']
        }
        del data

        return

    def InitBCsMod(self, args, attrs):
        BCsMod = bc_in_type(args['dpl_model']['phy_model']['ipedof'])
        BCsMod.setAttrs(attrs['dz'], attrs['zi'], attrs['zsoi'], attrs['sand'], attrs['clay'], attrs['om_frac'])
        # BCsMod.InitStates(forcing[...,0], forcing[...,1])

        self.models_Dict['BCsMod'] = BCsMod
        return

    def InitVars(self, args, dimsDict, paramsDict, attrs,params, BCsMod):
        rhiz_updates = F_init.setRhizlayers(nlevsoil= dimsDict['nlevsoil'],
                                            nlevrhiz= dimsDict['nlevrhiz'],
                                            zi_sisl = BCsMod.zi_sisl,
                                            dz_sisl = BCsMod.dz_sisl,
                                            aggmeth = args['dpl_model']['phy_model']['aggmeth'])

        rhiz_keys = ['map_r2s', 'map_r2s_sparse', 'zi_rhiz', 'dz_rhiz']
        self.mvars.update(dict(zip(rhiz_keys, rhiz_updates)))
        # ===============================================================================================================
        conn_updates = F_init.SetConnections(dimsDict['n_hypool_leaf'] , dimsDict['n_hypool_stem'],
                                             dimsDict['n_hypool_troot'], dimsDict['n_hypool_aroot'] ,
                                             dimsDict['nlevrhiz']      , dimsDict['nshell'],
                                             dimsDict['num_nodes']     , dimsDict['num_connections'],
                                             paramsDict['hydro_media_id'])

        conn_keys = ['conn_dn', 'conn_up', 'pm_node', 'node_layer', 'icnx_', 'inode_', 'mask_plant_media' ]
        self.mvars.update(dict(zip(conn_keys, conn_updates)))
        # ===============================================================================================================
        z_updates = F_init.InitPlantHydrNodes(n_hypool_leaf = dimsDict['n_hypool_leaf'],
                                              n_hypool_stem = dimsDict['n_hypool_stem'],
                                              zi_rhiz       = self.mvars['zi_rhiz'],
                                              prt_params    = params,
                                              pl_dbh        = attrs['pl_dbh'],
                                              pl_height     = attrs['pl_height'],
                                              )
        z_keys = ['z_upper_ag', 'z_lower_ag', 'z_node_troot', 'z_node_ag']
        self.mvars.update(dict(zip(z_keys, z_updates)))
        #===============================================================================================================
        lv_updates = F_init.InitPlantHydrLenVol(n_hypool_leaf    = dimsDict['n_hypool_leaf'],
                                                n_hypool_stem    = dimsDict['n_hypool_stem'],
                                                pl_sapwood       = attrs['pl_sapw_area'] ,
                                                pl_height        = attrs['pl_height'],
                                                pl_dbh           = attrs['pl_dbh'],
                                                prt_params       = params,
                                                zi_rhiz          = self.mvars['zi_rhiz'],
                                                dz_rhiz          = self.mvars['dz_rhiz'],
                                                    )
        lv_keys = ['v_ag', 'v_troot', 'l_aroot_layer_CC', 'v_aroot_layer']
        self.mvars.update(dict(zip(lv_keys, lv_updates)))
        #===============================================================================================================
        kmax_updates = F_init.InitPlantKmax(    n_hypool_leaf   = dimsDict['n_hypool_leaf'],
                                                n_hypool_stem   = dimsDict['n_hypool_stem'],
                                                pl_sapwood      = attrs['pl_sapw_area'],
                                                kmax_node_pft   = params['kmax_node'],
                                                p_taper         = params['p_taper'],
                                                rfrac_stem      = params['rfrac_stem'],
                                                hydr_r2s        = params['hydr_r2s'],
                                                l_aroot_layer_CC= self.mvars['l_aroot_layer_CC'],
                                                z_node_ag       = self.mvars['z_node_ag'],
                                                z_lower_ag      = self.mvars['z_lower_ag'],
                                                z_upper_ag      = self.mvars['z_upper_ag'],
                                                z_node_troot    = self.mvars['z_node_troot'],
                                                )
        kmax_keys = ['kmax_stem_upper', 'kmax_stem_lower', 'kmax_troot_upper', 'kmax_troot_lower', 'kmax_aroot_upper',
                     'kmax_aroot_lower', 'kmax_aroot_radial_in', 'kmax_aroot_radial_out']

        self.mvars.update(dict(zip(kmax_keys, kmax_updates)))
        # ==============================================================================================================
        rhiz_updates = F_init.InitSizeDepRhizVolLenCon(nshell               = dimsDict['nshell'],
                                                       hksat_sisl           = BCsMod.hksat_sisl,
                                                       dz_sisl              = BCsMod.dz_sisl,
                                                       si_code              = attrs['si_code'],
                                                       st_density           = attrs['st_density'],
                                                       l_aroot_layer_CC     = self.mvars['l_aroot_layer_CC'],
                                                       dz                   = self.mvars['dz_rhiz'],
                                                       map_r2s              = self.mvars['map_r2s_sparse'],
                                                       mean_type            = args['dpl_model']['phy_model']['mean_type'])
        rhiz_keys = ['l_aroot_layer', 'r_out_shell', 'r_node_shell', 'v_shell',
                     'kmax_upper_shell', 'kmax_lower_shell']

        self.mvars.update(dict(zip(rhiz_keys, rhiz_updates)))
        # ==============================================================================================================

        zv_updates = F.GetZVnode( n_hypool_aroot =  dimsDict['n_hypool_aroot'],
                                  n_hypool_troot =  dimsDict['n_hypool_troot'],
                                  num_nodes      =  dimsDict['num_nodes'],
                                  z_node_ag      =  self.mvars['z_node_ag'],
                                  v_ag           =  self.mvars['v_ag'],
                                  z_node_troot   =  self.mvars['z_node_troot'],
                                  v_troot        =  self.mvars['v_troot'],
                                  v_aroot_layer  =  self.mvars['v_aroot_layer'],
                                  v_shell        =  self.mvars['v_shell'],
                                  l_aroot_layer  =  self.mvars['l_aroot_layer'],
                                  l_aroot_layer_CC= self.mvars['l_aroot_layer_CC'],
                                  zi_rhiz        =  self.mvars['zi_rhiz'],
                                  dz_rhiz        =  self.mvars['dz_rhiz']
                                  )
        key_updates = ['z_node', 'v_node']
        self.mvars.update(dict(zip(key_updates, zv_updates)))

        kmax_dn, kmax_up = F.SetMaxCondConnections( dims                    = dimsDict,
                                                    l_aroot_layer_CC        = self.mvars['l_aroot_layer_CC'],
                                                    l_aroot_layer           = self.mvars['l_aroot_layer'],
                                                    kmax_stem_upper         = self.mvars['kmax_stem_upper'],
                                                    kmax_stem_lower         = self.mvars['kmax_stem_lower'],
                                                    kmax_troot_upper        = self.mvars['kmax_troot_upper'],
                                                    kmax_troot_lower        = self.mvars['kmax_troot_lower'],
                                                    kmax_aroot_upper        = self.mvars['kmax_aroot_upper'],
                                                    kmax_aroot_lower        = self.mvars['kmax_aroot_lower'],
                                                    kmax_aroot_radial_in    = self.mvars['kmax_aroot_radial_in'],
                                                    kmax_upper_shell        = self.mvars['kmax_upper_shell'],
                                                    kmax_lower_shell        = self.mvars['kmax_lower_shell'],
                                                    )
        self.mvars['kmax_up']   = kmax_up
        self.mvars['kmax_dn'] = kmax_dn
        return

    def InitWTFs(self, args, dimsDict, paramsDict, attrs, BCsMod):
        wrf_plant, wkf_plant = F_init.InitPlantWTF(  args['dpl_model']['phy_model']['plant_wrf_type'] ,
                                                     dimsDict['n_hypool_plant'],
                                                     dimsDict['n_hypool_aroot'],
                                                     dimsDict['nlevrhiz'],
                                                     dimsDict['numpft'],
                                                     paramsDict['vg_plant'],
                                                     paramsDict['tfs_wrf'],
                                                     paramsDict['tfs_wkf'],
                                                     paramsDict['tfs_stomata'],
                                                     paramsDict['hydro_media_name'],
                                                     paramsDict['hydro_media_id'],
                                                     paramsDict['rwccap'],
                                                     paramsDict['rwcft'],
                                                     attrs['pl_pft'],
                                                     self.mvars['pm_node'],
                                             )

        self.models_Dict['wrf_plant'], self.models_Dict['wkf_plant'] = wrf_plant, wkf_plant

        wrf_soil, wkf_soil = F_init.InitSoilWTF( args['dpl_model']['phy_model']['soil_wrf_type'] ,
                                                 args['dpl_model']['phy_model']['soil_wkf_type'],
                                                 args['dpl_model']['phy_model']['mean_type'],
                                                 dimsDict['nlevrhiz'],
                                                 dimsDict['numpft'],
                                                 paramsDict['vg_soil'],
                                                 BCsMod.dz_sisl,
                                                 BCsMod.watsat_sisl,
                                                 BCsMod.sucsat_sisl,
                                                 BCsMod.bsw_sisl,
                                                 attrs['pl_pft'],
                                                 self.mvars['map_r2s_sparse'])
        self.models_Dict['wrf_soil'], self.models_Dict['wkf_soil'] = wrf_soil, wkf_soil

        return

    def get_forcing(self,args, forcing_list, params, attrs):
        varLsT = args['dpl_model']['phy_model']['forcings']
        phys_forcing_LsT = []
        soil_forcing_LsT = []

        for ind, forcing in enumerate(forcing_list):
            forcing = torch.tensor(forcing, dtype =fates_r8 ).unsqueeze(0) #torch.float32, device = args['device']
            indices = [varLsT.index(item) for item in ['SM_0_kgm2', 'SM_1_kgm2', 'SM_2_kgm2', 'SM_3_kgm2', 'SM_4_kgm2']]
            h2osoi_liq    = forcing[..., indices]
            air_tempk     = forcing[..., varLsT.index('air_tempk')]
            RH            = forcing[..., varLsT.index('RH')]
            can_press     = forcing[..., varLsT.index('can_press')]
            can_o2_ppress = forcing[..., varLsT.index('can_o2_ppress')]
            co2_ppress    = forcing[..., varLsT.index('co2_ppress')]
            PARin  = forcing[..., varLsT.index('PARin')]
            wind_v = forcing[..., varLsT.index('ws')]
            wind_v = torch.where(wind_v == 0, 1.e-3, wind_v)  # ****************************************************
            kpft   = params['kpft'][[ind]].to("cpu")
            lai    = attrs['lai'][[ind]].to("cpu")
            cummulative_lai = attrs['cummulative_lai'][[ind]].to("cpu")

            # GET leaflayer photosynthetic active radiation wm-2
            # nsites, ncohorts,nt, nll
            parsun_In = F.getPARlsl_In(kpft=kpft.unsqueeze(dim=-1).unsqueeze(dim=-1),
                                       PARin=PARin.unsqueeze(dim=1).unsqueeze(dim=-1),
                                       cummulative_lai=cummulative_lai.unsqueeze(dim=2))

            parlsl_abs = (parsun_In[..., :-1] - parsun_In[..., 1:]) * (1 - Fref / (Fref + Fabs))
            # ================================================================================
            # GET the boundary layer resistance s/m
            rb = 100.0 * (math.sqrt(dleaf) / torch.sqrt(wind_v))
            # ===============================================================================
            # GET the air vapor pressure (pa)
            air_vpress = F.QSat(air_tempk) * RH / 100
            # ===============================================================================
            # GET some CanopyGasParameters
            kc25 = (mm_kc25_umol_per_mol / umol_per_mol) * can_press
            ko25 = (mm_ko25_mmol_per_mol / mmol_per_mol) * can_press
            sco  = 0.5 * 0.209 / (co2_cpoint_umol_per_mol / umol_per_mol)
            cp25 = 0.5 * can_o2_ppress / sco
            # ===============================================================================
            # GET boundary layer conductance (gb_mol) and conversion term (cf)
            cf = can_press / (rgas_J_K_kmol * air_tempk) * umol_per_kmol
            gb_mol = (1.0 / rb) * cf
            # ===============================================================================
            # GET PAR absorbed by PS II (umol photons/m**2/s)
            # ===============================================
            qabs = torch.where(lai.unsqueeze(dim=2) > 0.0,  # unsqueeze to account for time dimension
                               parlsl_abs * photon_to_e * (1. - fnps) * wm2_to_umolm2s,
                               0.0)
            # ===============================================================================
            # GET soil ice content (assume zero for now across different soil layers
            # ===============================================
            h2osoi_ice = torch.zeros_like(h2osoi_liq)
            # ===============================================================================
            nsites, ncohorts, nt, nlayers = parlsl_abs.shape
            phys_forcing = torch.stack(
                [kc25, ko25, cp25, cf, air_vpress, air_tempk, can_press, can_o2_ppress, co2_ppress, gb_mol, rb], dim=-1)
            phys_forcing = phys_forcing.unsqueeze(dim=1).unsqueeze(dim=3).expand(-1, ncohorts, -1, nlayers, -1)
            phys_forcing = torch.cat((phys_forcing, parlsl_abs.unsqueeze(-1), qabs.unsqueeze(dim=-1)), dim=-1)

            soil_forcing = torch.stack([h2osoi_liq, h2osoi_ice], dim=-1)

            phys_forcing = phys_forcing.squeeze(0).permute(1,0,2,3) # nt, ncohorts, nll, nf
            soil_forcing = soil_forcing.squeeze(0)                  # nt, nlevrhiz, nf
            # return phys_forcing, soil_forcing
            phys_forcing_LsT.append(phys_forcing)
            soil_forcing_LsT.append(soil_forcing)
        return {"physical": phys_forcing_LsT, "soil":soil_forcing_LsT}

    def get_attrs(self, attrs):
        return self.get_final_output(attrs,  ['pl_crown_area', 'lai'], ['pl_crown_area'], attrs['lai'], do_unsqueeze= True)

    def get_params(self,params):
        keys = ['vcmax25top_ft', 'medlyn_slope' , 'stomatal_intercept','vcmaxha', 'vcmaxse', 'vcmaxhd','jmaxha' , 'jmaxhd' , 'jmaxse']
        return self.get_final_output(params, keys + ['nscaler'], keys, params['nscaler'], do_unsqueeze= True)

    def get_mvars(self):
        # List of keys to extract
        keys_to_extract =  ['v_node' ,'z_node','z_node_ag','z_node_troot','l_aroot_layer_CC',
                            'kmax_aroot_lower','kmax_aroot_radial_in','kmax_aroot_radial_out','kmax_up','kmax_dn',
                            'dz_rhiz', 'zi_rhiz']

        return {key: self.mvars[key] for key in keys_to_extract if key in self.mvars}

    def get_mconn(self):
        # List of keys to extract
        keys_to_extract = ['pm_node', 'mask_plant_media','icnx_', 'inode_', 'conn_dn', 'conn_up', 'map_r2s_sparse']

        return {key: self.mvars[key] for key in keys_to_extract if key in self.mvars}

    def get_final_output(self, dict, keys_to_extract, keys_to_unsqueeze=None, target=None, do_unsqueeze=False):
        if do_unsqueeze:
            # Unsqueeze tensors for specified keys in-place
            for key in keys_to_unsqueeze:
                dict[key] = dict[key].unsqueeze(-1).expand_as(target)

        out = torch.stack([dict[key] for key in keys_to_extract], dim=-1)
        return out
#=======================================================================================================================



















