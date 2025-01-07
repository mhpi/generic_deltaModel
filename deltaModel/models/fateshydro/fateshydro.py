import sys
sys.path.append("../deltaModel/models/fateshydro")


import time
import torch.nn
import numpy as np
from tqdm import tqdm
from Init       import InitFuncs as F_init
from data.utils import createTensor
from physical.biogeophys import Functionals as F
from conf.Constants             import dens_fresh_liquid_water as denh2o, theta_psii, rsmax0, fates_r8, mpa_per_pa, grav_earth, nearzero
from solver.solverMod    import *

class fateshydro(torch.nn.Module):
    def __init__(self,args, device):
        super(fateshydro, self).__init__()
        self.args       = args
        self.settings   = args['settings']
        self.J          = Jacobian(mtd=args['settings']['solver'])
        self.mstates    = {}
        self.learnable_param_count = args['dynamic_params']['num']


        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, dataDict, params= None):
        dy_params = params = None
        # scaled_params   = self.scale_params(params, self.args['dynamic_params']['ranges'])

        self.forcing    = dataDict['x_phy']
        self.soilforcing= dataDict['x_soil']
        self.params     = dataDict['p_phy']
        self.attributes = dataDict['c_phy']
        self.mvars      = dataDict['v_phy']
        self.mconns     = dataDict['mconns']
        nly             = dataDict['dimsDict']['nlayers']
        modelsDict      = dataDict['aux']
        hydro_media_id  = dataDict['paramsDict']['hydro_media_id']

        wrf_plant = modelsDict['wrf_plant']
        wkf_plant = modelsDict['wkf_plant']
        wrf_soil  = modelsDict['wrf_soil']
        wkf_soil  = modelsDict['wkf_soil']
        dtime     = self.args['dtime']
        n_steps   = self.forcing.shape[0]

        self.out = {"sapflow": [], 'wb_err_plant': []}  # , 'psi': [], "kmax_dn": []

        pbar = tqdm(total=n_steps, desc="Progress")
        for t in range(n_steps):

            time.sleep(0.01)
            #if t % 10 == 0:
            pbar.update(1)

            # get time step forcing
            c_forcing = self.forcing[t]
            dy_params = None #scaled_params[t]

            # update the states (th_plant, btran, kmax_dn, th_node_init,h2osoi_liqvol_shell)
            self.UpdateStates(hydro_media_id, modelsDict['BCsMod'], modelsDict['wrf_soil'],
                              modelsDict['wrf_plant'], modelsDict['wkf_plant'], dataDict['dimsDict'], t)

            # First Solve
            ci_init, veg_tempk_init, th_init    = self.get_xinit(c_forcing,t)
            f     = nonlinearsolver(self,c_forcing,wrf_plant, wkf_plant, wrf_soil, wkf_soil, hydro_media_id, dtime, nly, mtd ="solve", p = dy_params)
            vG    = tensorNewton(f, self.J, settings= self.settings, mtd=self.settings['solver'])
            x_sol = vG(torch.concat([ci_init, veg_tempk_init, th_init], dim = -1))
            # vG    = tensorNewton.apply
            # x_sol = vG(torch.concat([ci_init, veg_tempk_init, th_init], dim = -1), f, self.J, self.settings['solver'], self.settings)

            # ========================================
            # ========================================
            ci, veg_tempk, th_node = x_sol[...,0:nly], x_sol[...,nly:2*nly], x_sol[...,2*nly:]
            with torch.no_grad():
                qcanopy, kmax_dn, sapflow = self.solve_for_Q_th(ci,veg_tempk, th_node, c_forcing,
                                                                wrf_plant, wkf_plant, wrf_soil, wkf_soil,
                                                                hydro_media_id, dtime,mtd = 1 , p=dy_params)

                th_node_init   = self.mstates['th_node_init']
                v_node         = self.mvars['v_node']
                dth_node       = th_node - th_node_init
                th_plant       = self.mstates['th_plant'] + dth_node[..., self.mconns['mask_plant_media']]

                self.out['wb_err_plant'].append(self.get_plant_err(th_node_init, th_node, v_node, qcanopy, dtime).abs().max().item())
                self.out['sapflow'].append(sapflow.clone())

                psi_plant, ftc_plant, btran = self.solve_for_btran(wrf_plant, wkf_plant, th_plant)
                updatestates = [qcanopy  , kmax_dn  , ci  , veg_tempk  , th_node  , th_plant , btran  ]
                updatekeys   = ['qcanopy', 'kmax_dn', 'ci', 'veg_tempk', 'th_node','th_plant', 'btran']
                self.mstates.update(dict(zip(updatekeys, updatestates)))


        print(np.abs(np.asarray(self.out['wb_err_plant'])).max())
        # assert torch.all(wb_err_plant < 1e-10)
        # pbar.close()
        return {'sapf': torch.stack(self.out['sapflow'], dim = 0)}

    def solve_for_Q_th(self,
                       ci,
                       veg_tempk,
                       th_node,
                       c_forcing,
                       wrf_plant, wkf_plant, wrf_soil, wkf_soil,
                       hydro_media_id,
                       dtime=None, mtd=0, p=None):

        # Forcing variables
        kc25            = c_forcing[..., 0]
        ko25            = c_forcing[..., 1]
        cp25            = c_forcing[..., 2]
        cf              = c_forcing[..., 3]
        air_vpress      = c_forcing[..., 4]
        air_tempk       = c_forcing[..., 5]
        can_press       = c_forcing[..., 6]
        can_o2_ppress   = c_forcing[..., 7]
        co2_ppress      = c_forcing[..., 8]
        gb_mol          = c_forcing[..., 9]
        rb              = c_forcing[..., 10]
        parlsl_abs      = c_forcing[..., 11]
        qabs            = c_forcing[..., 12]

        # parameters
        vcmax25top_ft       = self.params[..., 0]
        medlyn_slope        = self.params[..., 1]
        stomatal_intercept  = self.params[..., 2]
        vcmaxha             = self.params[..., 3]
        vcmaxse             = self.params[..., 4]
        vcmaxhd             = self.params[..., 5]
        jmaxha              = self.params[..., 6]
        jmaxhd              = self.params[..., 7]
        jmaxse              = self.params[..., 8]
        nscaler             = self.params[..., 9]

        # attributes
        c_area  = self.attributes[..., 0]
        lai_lsl = self.attributes[..., 1]

        # states
        btran           = self.mstates['btran'].unsqueeze(dim=-1).clone()
        kmax_dn         = self.mstates['kmax_dn'].clone()
        th_node_init    = self.mstates['th_node_init'].clone()

        # variables
        v_node                  = self.mvars['v_node']
        z_node                  = self.mvars['z_node']
        kmax_aroot_lower        = self.mvars['kmax_aroot_lower']
        kmax_aroot_radial_in    = self.mvars['kmax_aroot_radial_in']
        kmax_aroot_radial_out   = self.mvars['kmax_aroot_radial_out']
        kmax_up                 = self.mvars['kmax_up']

        # connections
        pm_node = self.mconns['pm_node']
        icnx_   = self.mconns['icnx_']
        inode_  = self.mconns['inode_']
        conn_dn = self.mconns['conn_dn']
        conn_up = self.mconns['conn_up']
        mask_plant_media = self.mconns['mask_plant_media']

        # ==============================================================================================================
        veg_esat = F.QSat(veg_tempk)

        # GET the canopy gas parameters
        # ==============================
        mm_kco2, mm_ko2, co2_cpoint, ceair = F.GetCanopyGasParameters(veg_tempk, air_vpress, veg_esat, kc25, ko25, cp25)

        # GET the biophysical rates
        # ==========================
        # GET the top of canopy biophysical rates
        # ========================================
        jmax25top_ft, co2_rcurve_islope25top_ft, lmr25top_ft = F.GetCanopytopRates(vcmax25top_ft)
        vcmax, jmax, co2_rcurve_islope = F.LeafLayerBiophysicalRates(parlsl_abs, vcmax25top_ft, jmax25top_ft,
                                                                     co2_rcurve_islope25top_ft,
                                                                     vcmaxha, vcmaxhd, vcmaxse, jmaxha, jmaxhd, jmaxse,
                                                                     nscaler, veg_tempk, btran)
        # ==========================
        lmr = F.LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler, veg_tempk)
        # GET medlyn's stomatal intercept
        # ===============================
        stomatal_intercept_btran = torch.max(cf / rsmax0, (stomatal_intercept * btran))

        # GET electron transport rate (umol electrons/m**2/s)
        # ====================================================
        je = F.quadratic_min(theta_psii, -(qabs + jmax), qabs * jmax)

        # GET leaflayer net photosynthesis
        # ===============================
        anet = F.getleaflayerAn(ci, je, co2_cpoint, mm_kco2, mm_ko2, vcmax, lmr, can_o2_ppress, parlsl_abs, lai_lsl)

        # GET CO2 partial pressure at and inside the leaf
        # ===============================================
        can_co2_ppress, leaf_co2_ppress = F.getCANC02press(self.settings['at_leaf'], co2_ppress, gb_mol, anet,
                                                           can_press)

        # GET the stomatal conductance
        # ============================
        gs_mol = F.getleaflayergs(anet, leaf_co2_ppress, can_press, veg_esat, ceair, stomatal_intercept_btran,
                                  medlyn_slope, gb_mol)

        # ============================
        qtop = F.getqtop_layer(gs_mol, cf, c_area, lai_lsl, rb, veg_esat, ceair, air_tempk, can_press)
        qcanopy = qtop.sum(dim=-1)

        # ============================
        if p is not None:
            dy_k = p[..., 0:1].clone().unsqueeze(dim=-1)  # for connection dim #.unsqueeze(1) # for cohort dim#
            # wkf_plant_params = self.updateparams(p, self.args)
            # wkf_plant['plant_media'].updateAttrs(wkf_plant_params)
        else:
            dy_k = None
        h_node, psi_node, ftc_node = F.GethPsiftcnode(z_node, pm_node, mask_plant_media,
                                                      wrf_plant, wkf_plant, wrf_soil, wkf_soil, th_node, hydro_media_id)

        # ============================
        kmax_dn = F.UpdatekmaxDn(kmax_dn, kmax_aroot_lower,
                                 kmax_aroot_radial_in, kmax_aroot_radial_out,
                                 h_node, icnx_, inode_)

        # ============================
        q_flux = F.GetQflux(conn_dn, conn_up, kmax_dn, kmax_up, h_node, ftc_node)
        q_flux = q_flux * dy_k if dy_k is not None else q_flux

       # q_flux = F.GetQflux(conn_dn, conn_up, kmax_dn, kmax_up, h_node, ftc_node)
       # q_flux[...,1] = q_flux[...,1] * pDyn
       # MULTI
       # learn for plant connections only #
       # q_flux = F.GetQflux(conn_dn, conn_up, kmax_dn, kmax_up, h_node, ftc_node)
       # mask_plant_conn = mask_plant_media.roll(-1)[:-1]
       # q_flux[..., mask_plant_conn] = q_flux[..., mask_plant_conn] * pDyn

        if mtd == 0 or mtd == "solve":
            f1 = F.GetResidual_ci(ci, anet, gs_mol, can_co2_ppress, can_press, gb_mol, co2_ppress)  # c3c4_path_index
            f2 = F.GetResidual_Tleaf(veg_tempk, air_tempk, parlsl_abs, 1 / gb_mol, 1.0 / gs_mol, veg_esat, ceair,
                                     can_press)
            f3 = F.GetResidual_th(qcanopy, q_flux, conn_dn, conn_up, v_node, th_node, th_node_init, dtime)
            f  = torch.concat([f1, f2, f3], dim=-1)#.squeeze(0)

            return f
        else:
            stem_index = torch.where(pm_node == hydro_media_id['stem_p_media'])[0].item()
            sapflow = q_flux[:, :, stem_index] * dtime
            return qcanopy, kmax_dn, sapflow

    def solve_for_btran(self, wrf_plant, wkf_plant, th_plant):
        psi_plant, ftc_plant = F.UpdatePlantPsiFTCFromTheta(th_plant=th_plant,
                                                            wrf_plant=wrf_plant,
                                                            wkf_plant=wkf_plant)

        btran = wkf_plant['stomata_p_media'].ftc_from_psi(psi_plant[..., 0])
        return psi_plant, ftc_plant, btran

    def get_xinit(self, c_forcing, t):
        air_tempk = c_forcing[..., 5]
        co2_ppress = c_forcing[..., 8]

        veg_tempk_init = F.get_guess_tleaf(air_tempk)

        if t == 0:
            ci_init = F.get_guess_ci(co2_ppress)  # c3c4_path_index
            th_init = self.mstates["th_node_init"].clone().detach()
        else:
            ci_init = self.mstates['ci'].clone().detach()
            th_init = self.mstates['th_node'].clone().detach()

        # ci_init.requires_grad_(True)
        # veg_tempk_init.requires_grad_(True)
        # th_init.requires_grad_(True)

        return ci_init, veg_tempk_init, th_init

    def UpdateStates(self, hydro_media_id, BCMod,wrf_soil, wrf_plant, wkf_plant, dimsDict, t):

        BCMod.UpdateStates(self.soilforcing[...,0], self.soilforcing[...,1], t, BCMod.watsat_sisl, BCMod.dz_sisl)
        h2osoi_liqvol_shell = F_init.UpdateSoilHydStates(dz_sisl        = BCMod.dz_sisl,
                                                         eff_porosity_sl= BCMod.eff_porosity_sl,
                                                         h2o_liq_sisl   = BCMod.h2o_liq_sisl,
                                                         dz_rhiz        = self.mvars['dz_rhiz'],
                                                         map_r2s        = self.mconns['map_r2s_sparse'],
                                                         nshell         = dimsDict['nshell'],
                                                         mean_type      = self.args['mean_type'])

        #===================================================================================
        if t == 0:
            Hydstates_updates = F_init.InitPlantHydStates(init_mode             = self.args['init_mode'],
                                                          h2osoi_liqvol_shell   = h2osoi_liqvol_shell,
                                                          l_aroot_layer_CC      = self.mvars['l_aroot_layer_CC'],
                                                          z_node_troot          = self.mvars['z_node_troot'],
                                                          z_node_ag             = self.mvars['z_node_ag'],
                                                          zi_rhiz               = self.mvars['zi_rhiz'],
                                                          dz_rhiz               = self.mvars['dz_rhiz'],
                                                          wrf_soil              = wrf_soil,
                                                          wrf_plant             = wrf_plant,
                                                          wkf_plant             = wkf_plant)

            self.mstates['th_plant']= Hydstates_updates[0]
            self.mstates['btran']   = Hydstates_updates[1]
            self.mstates['kmax_dn'] = self.mvars['kmax_dn'].clone()
            self.mstates["th_node_init"] = createTensor((*self.mstates['btran'].shape, dimsDict['num_nodes']),
                                                        dtype=fates_r8)
            # ===============================================================================================================
        th_node_init = F.Gethnode(mask_plant_media    = self.mconns['mask_plant_media'],
                                  th_node_init        = self.mstates['th_node_init'],
                                  th_plant            = self.mstates['th_plant'],
                                  h2osoi_liqvol_shell = h2osoi_liqvol_shell,
                                  pm_node             = self.mconns['pm_node'],
                                  hydr_media_id       = hydro_media_id)

        self.mstates["th_node_init"]        = th_node_init
        self.mstates["h2osoi_liqvol_shell"] = h2osoi_liqvol_shell

        return


    def updateparams(self,params, args):
        updated_params = {key: params[:, 1:].unsqueeze(dim=1).clone() for i, key in
                          enumerate(args['dynamic_params']['Fateshydro'])}

        return updated_params

    def get_plant_err(self, th_node_init, th_node, v_node, qcanopy, dtime):
        #   ! Total water mass in the plant at the beginning of this solve [kg h2o]
        w_tot_beg = (th_node_init * v_node).sum(dim=-1) * denh2o
        # ! Total water mass in the plant at the end of this solve [kg h2o]
        w_tot_end = (th_node * v_node).sum(dim=-1) * denh2o
        # ! Mass error (flux - change) [kg/m2]
        wb_err_plant = (qcanopy * dtime) - (w_tot_beg - w_tot_end)
        return wb_err_plant

    def scale_params(self, p, p_ranges):
        p_out = p.clone()
        for i in range(p_out.shape[-1]):
            p_out[..., i] = p[..., i] * (p_ranges['max'][i] - p_ranges['min'][i]) + p_ranges['min'][i]

        return p_out





    # def forward_nsteps(self, dtime, modelsDict, params):
    #     psi_plant = torch.stack(self.out['psi'])
    #     kmax_dn   = torch.stack(self.out['kmax_dn'])
    #
    #     mask_plant_media = self.mconns['mask_plant_media']
    #     mask_plant_conn  = mask_plant_media.roll(-1)[:-1]
    #     conn_dn          = self.mconns['conn_dn'][mask_plant_conn]
    #
    #     kmax_up = self.mvars['kmax_up'][...,mask_plant_conn].unsqueeze(dim=0)
    #     kmax_dn = kmax_dn[..., mask_plant_conn]
    #     z_plant = self.mvars['z_node'][...,mask_plant_media].unsqueeze(dim=0)
    #     h_plant = mpa_per_pa * denh2o * grav_earth * z_plant + psi_plant
    #
    #     p50, avuln = params[...,1:].unsqueeze(dim=2), modelsDict['wkf_plant']['plant_media'].avuln# modelsDict['wkf_plant']['plant_media'].p50, modelsDict['wkf_plant']['plant_media'].avuln#params[...,0:8].unsqueeze(dim=2), params[...,8:16].unsqueeze(dim=2)
    #
    #     psi_eff   = torch.clamp(psi_plant, max = -nearzero)
    #     ftc_plant = torch.clamp( 1.0 / (1.0 + (psi_eff / p50) ** avuln), min =  0.00001)
    #
    #     idxs_dn = conn_dn - 1
    #
    #     h_node_dn = h_plant[..., idxs_dn]
    #     h_node_up = h_plant[..., 1:]
    #     ftc_node_dn = ftc_plant[..., idxs_dn]
    #     ftc_node_up = ftc_plant[..., 1:]
    #     k_eff = F.GetKAndDKDPsi(kmax_dn,
    #                             kmax_up,
    #                             h_node_dn,
    #                             h_node_up,
    #                             ftc_node_dn,
    #                             ftc_node_up,
    #                             do_upstream_k = True)
    #     q_flux = k_eff * (h_node_up - h_node_dn) * dtime * params[...,0:1].unsqueeze(-1)#params.unsqueeze(-1)  #.unsqueeze(2)
    #     sapflow = q_flux[...,1]
    #
    #     # q_flux = k_eff * (h_node_up - h_node_dn) * dtime
    #     # sapflow = q_flux[...,1]  * params
    #     # psi_node  = torch.stack(self.out['psi'])
    #     # ftc_node  = torch.stack(self.out['ftc'])
    #     #
    #     # kmax_dn   = torch.stack(self.out['kmax_dn'])
    #     # conn_dn = self.mconns['conn_dn']
    #     # conn_up = self.mconns['conn_up']
    #     #
    #     # kmax_up = self.mvars['kmax_up'].unsqueeze(dim=0)
    #     # z_node = self.mvars['z_node'].unsqueeze(dim=0)
    #     # h_node = mpa_per_pa * denh2o * grav_earth * z_node + psi_node
    #     #
    #     #
    #     # idxs_dn = conn_dn - 1
    #     # idxs_up = conn_up - 1
    #     # h_node_dn = h_node[..., idxs_dn]
    #     # h_node_up = h_node[..., idxs_up]
    #     # ftc_node_dn = ftc_node[..., idxs_dn]
    #     # ftc_node_up = ftc_node[..., idxs_up]
    #     # k_eff = F.GetKAndDKDPsi(kmax_dn,
    #     #                       kmax_up,
    #     #                       h_node_dn,
    #     #                       h_node_up,
    #     #                       ftc_node_dn,
    #     #                       ftc_node_up,
    #     #                       do_upstream_k=True)
    #     #
    #     # q_flux = k_eff * (h_node_up - h_node_dn) * dtime * params.unsqueeze(dim=2)
    #     # sapflow = q_flux[..., 1]
    #     return sapflow









