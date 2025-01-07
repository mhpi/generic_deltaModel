import numpy as np

from physical.biogeophys import Functionals as F
from physical.biogeochem import AllometryMod as F_allom
from physical.biogeophys.HydroWTMod import *

from conf.Constants import *
from conf.Constants import dens_fresh_liquid_water as denh2o

from data.utils import createTensor
#=======================================================================================================================
def InitPlantHydrNodes(n_hypool_leaf,n_hypool_stem,zi_rhiz,
                       prt_params,
                       pl_dbh,
                       pl_height,
                       ):
    """
    Calculate the nodal heights critical to hydraulics in the plant.

    :param ccohort: ccohort object with plant information.
    :param ft: Plant functional type index.
    :param plant_height: Height of the plant [m].
    :param csite_hydr: Site hydrology object.
    """

    n_hypool_ag      = n_hypool_leaf + n_hypool_stem
    z_lower_ag       = createTensor((*pl_dbh.shape, n_hypool_ag), fill_nan=False, dtype=fates_r8)
    z_upper_ag       = createTensor((*pl_dbh.shape, n_hypool_ag), fill_nan=False, dtype=fates_r8)
    z_node_ag        = createTensor((*pl_dbh.shape, n_hypool_ag), fill_nan=False, dtype=fates_r8)


    # Crown Nodes
    crown_depth = torch.clamp(pl_height, max= 0.1)
    dz_canopy   = crown_depth / n_hypool_leaf
    for k in range(n_hypool_leaf):
        z_lower_ag[:,:,k] = pl_height - dz_canopy * (k + 1)
        z_node_ag[:,:,k]  = z_lower_ag[:,:,k] + 0.5 * dz_canopy
        z_upper_ag[:,:,k] = z_lower_ag[:,:,k] + dz_canopy

        # Stem Nodes
    z_stem = pl_height - crown_depth
    dz_stem = z_stem / n_hypool_stem
    for k in range(n_hypool_leaf, n_hypool_ag):
        index = k - n_hypool_leaf
        z_upper_ag[:,:,k] = (n_hypool_stem - index) * dz_stem
        z_node_ag[:,:,k]  = z_upper_ag[:,:,k] - 0.5 * dz_stem
        z_lower_ag[:,:,k] = z_upper_ag[:,:, k] - dz_stem

    # Maximum Rooting Depth
    # sites_pft, sites_dbh, z_max_soil
    z_fr = F.MaximumRootingDepth(sites_dbh = pl_dbh,
                                 z_max_soil= zi_rhiz[:,-1],
                                 zroot_max_dbh = prt_params['zroot_max_dbh'],
                                 zroot_min_dbh = prt_params['zroot_min_dbh'],
                                 zroot_max_z   = prt_params['zroot_max_z'],
                                 zroot_min_z   = prt_params['zroot_min_z'],
                                 zroot_k       = prt_params['zroot_k'],
                                 )#.unsqueeze(dim=-1) already done in the function

    z_cumul_rf = F.bisect_rootfr(prt_params['roota'],
                                 prt_params['rootb'],
                                 z_max   = z_fr,
                                 lower   = torch.zeros_like(z_fr)+0.0,
                                 upper   = torch.zeros_like(z_fr)+1.E10,
                                 xtol    = 0.001,
                                 ytol    = 0.001,
                                 crootfr = 0.5)

    mask = z_cumul_rf > zi_rhiz[:, -1].unsqueeze(dim= -1) # expand in ncohorts dimension)
    if mask.any():
        # This would be replaced with appropriate error handling in Python
        raise ValueError('Error: z_cumul_rf > zi_rhiz[nlevrhiz]?')

    z_cumul_rf = torch.minimum(z_cumul_rf,
                               torch.abs(zi_rhiz[:,-1]).unsqueeze(dim=-1))
                               # expand in ncohorts dimension

    z_node_troot   = -z_cumul_rf

    # outDict = {
    #     'z_upper_ag'  : z_upper_ag,
    #     'z_lower_ag'  : z_lower_ag,
    #     'z_node_troot': z_node_troot,
    #     'z_node_ag'   : z_node_ag,
    # }
    return  z_upper_ag, z_lower_ag, z_node_troot, z_node_ag

########################################################################################################################
########################################################################################################################
def InitPlantHydrLenVol(  n_hypool_leaf, n_hypool_stem,
                          pl_sapwood,
                          pl_height,
                          pl_dbh,
                          prt_params,
                          zi_rhiz,
                          dz_rhiz,
                          ):
    # assume a_sapwood is an input
    # required to accomodate for:
    # 1. a_sapwood
    # 2. commented bleaf, sites_hydr.v_ag[:,:,:n_hypool_leaf] , bsap_allom
    # 2. woody_bg_c [ft_id] = f(sapw_c, struct_c)
    # 3. fnrt_c[ft_id]

    # ! We allow the transporting root to donate a fraction of its volume to the absorbing
    # ! roots to help mitigate numerical issues due to very small volumes. This is the
    # ! fraction the transporting roots donate to those layers
    # ! Fraction of maximum leaf carbon that we set as our lower cap on leaf volume
    # ! The lower cap on trimming function used to estimate maximum leaf carbon

    n_hypool_ag   = n_hypool_leaf + n_hypool_stem
    nlevrhiz     = zi_rhiz.shape[-1]

    v_ag             = createTensor((*pl_dbh.shape, n_hypool_ag), fill_nan=False, dtype=fates_r8)
    l_aroot_layer_CC = createTensor((*pl_dbh.shape, nlevrhiz), fill_nan=True, dtype=fates_r8)
    v_aroot_layer    = createTensor((*pl_dbh.shape, nlevrhiz), fill_nan=False, dtype=fates_r8)


    t2aroot_vol_donate_frac = 0.65
    l2sap_vol_donate_frac = 0.5
    min_leaf_frac = 0.1

    sla = prt_params['slatop'] * cm2_per_m2
    # ! empirical regression data from leaves at Caxiuana (~ 8 spp)
    denleaf = -2.3231 * sla / prt_params['c2b'] + 781.899
    ##########################################################################################################
    # Assumed crown damage = 1, canopy trim = max(1, min_trim)
    leaf_c_target, _ = F_allom.bleaf(d=pl_dbh,
                                     prt_params=prt_params,
                                     crowndamage=1,
                                     canopy_trim=max(1.0, min_trim),
                                     elongf_leaf=1)
    # Assumed leaf_c = leaf_c_target
    leaf_c = leaf_c_target

    # Assumed crown damage = 1, canopy trim = 1.0, elongf_Stem = 1.0
    sapw_c_target, _ = F_allom.bsap_allom(d = pl_dbh,
                                          h = pl_height,
                                          prt_params =prt_params,
                                          crowndamage=1.0,
                                          canopy_trim=1.0,
                                          elongf_stem=1.0)
    # Assumed sapw_c = sapw_c_target
    sapw_c = sapw_c_target

    # Assumed canopy trim = 1.0, l2fr =prt_params['l2fr'], elongf_fnrt = 1.0
    fnrt_c, _ = F_allom.bfineroot(d          =pl_dbh,
                                  prt_params =prt_params,
                                  canopy_trim=1.0,
                                  l2fr       =prt_params['l2fr'],
                                  elongf_fnrt=1.0)

    # Assumed crown damage = 1, elongf_Stem = 1.0
    agw_c, _ = F_allom.bagw_allom(d=pl_dbh,
                                  h= pl_height,
                                  prt_params=prt_params,
                                  crowndamage=1.0,
                                  elongf_stem=1.0)

    # Assumed  elongf_Stem = 1.0
    bgw_c, _ = F_allom.bbgw_allom(d=pl_dbh,
                                  h = pl_height,
                                  prt_params=prt_params,
                                  elongf_stem=1.0)

    struct_c, _ = F_allom.bdead_allom(prt_params=prt_params,
                                      bagw=agw_c,
                                      bbgw=bgw_c,
                                      bsap=sapw_c)
    ##########################################################################################################

    crown_depth = torch.clamp(pl_height, max=0.1)
    z_stem      = pl_height - crown_depth
    v_sapwood   = pl_sapwood * z_stem

    mask = prt_params['woody'] == 1
    mask = mask.bool()

    nleavs = n_hypool_leaf
    ags    = n_hypool_ag

    v_ag[:, :, : nleavs] = (torch.max(leaf_c,
                                    min_leaf_frac * leaf_c_target) * prt_params['c2b'] / denleaf / nleavs).unsqueeze(dim=-1)

    v_leaf_donate = v_ag[:, :, :nleavs] * l2sap_vol_donate_frac
    v_ag[:, :, :nleavs] = torch.where(~mask.unsqueeze(-1),
                                      v_ag[:, :, :nleavs] - v_leaf_donate,
                                      v_ag[:, :, :nleavs])

    v_ag[:, :, nleavs:ags] = torch.where(~mask,
                                         (v_sapwood + torch.sum(v_leaf_donate, dim=-1)) / n_hypool_stem,
                                         v_sapwood / n_hypool_stem).unsqueeze(-1)

    #     ! Get the target, or rather, maximum leaf carrying capacity of plant
    #     ! Lets also avoid super-low targets that have very low trimming functions
    #
    #     call bleaf(ccohort%dbh,ccohort%pft,ccohort%crowndamage, &
    #          max(ccohort%canopy_trim,min_trim),ccohort%efleaf_coh, leaf_c_target)
    #
    #     if( (ccohort%status_coh == leaves_on) .or. ccohort_hydr%is_newly_recruited ) then
    #        ccohort_hydr%v_ag(1:n_hypool_leaf) = max(leaf_c,min_leaf_frac*leaf_c_target) * &
    #             prt_params%c2b(ft) / denleaf/ real(n_hypool_leaf,r8)
    #     end if
    #
    #     ! Step sapwood volume
    #     ! -----------------------------------------------------------------------------------
    #
    #     ! BOC...may be needed for testing/comparison w/ v_sapwood
    #     ! kg  / ( g cm-3 * cm3/m3 * kg/g ) -> m3
    #     ! v_stem       = c_stem_biom / (prt_params%wood_density(ft) * kg_per_g * cm3_per_m3 )
    #
    #     ! calculate the sapwood cross-sectional area
    #     call bsap_allom(ccohort%dbh,ccohort%pft,ccohort%crowndamage, &
    #          ccohort%canopy_trim, ccohort%efstem_coh, a_sapwood_target,sapw_c_target)
    #
    #     ! uncomment this if you want to use
    #     ! the actual sapwood, which may be lower than target due to branchfall.
    #     a_sapwood = a_sapwood_target  ! * sapw_c / sapw_c_target

    woody_bg_c = (1.0 - prt_params['agb_frac']) * (sapw_c + struct_c)
    v_troot = woody_bg_c * prt_params['c2b'] / (prt_params['wood_density'] * kg_per_g * cm3_per_m3)

    # ! Estimate absorbing root total length (all layers)
    # ! SRL is in m/g
    # ! [m] = [kgC]*1000[g/kg]*[kg/kgC]*[m/g]
    l_aroot_tot = fnrt_c * g_per_kg * prt_params['c2b'] * prt_params['hydr_srl']
    # ! Estimate absorbing root volume (all layers)
    # ! ------------------------------------------------------------------------------
    v_aroot_tot = pi_const * (prt_params['hydr_r2s'] ** 2.0) * l_aroot_tot
    v_troot     = 0.5 * (v_troot + v_aroot_tot)

    # sites_pft, sites_dbh, z_max_soil
    z_fr = F.MaximumRootingDepth(sites_dbh      =pl_dbh,
                                 z_max_soil     =zi_rhiz[:, -1],
                                 zroot_max_dbh  =prt_params['zroot_max_dbh'],
                                 zroot_min_dbh  =prt_params['zroot_min_dbh'],
                                 zroot_max_z    =prt_params['zroot_max_z'],
                                 zroot_min_z    =prt_params['zroot_min_z'],
                                 zroot_k        =prt_params['zroot_k'],
                                 )#.unsqueeze(dim=-1) already done in the function

    for j in range(nlevrhiz):
        rootfr1 = F.zeng2001_crootfr(prt_params['roota'],
                                     prt_params['rootb'],
                                     zi_rhiz[:, j].unsqueeze(dim=-1), z_fr)

        rootfr2 = F.zeng2001_crootfr(prt_params['roota'],
                                     prt_params['rootb'],
                                     (zi_rhiz[:, j] - dz_rhiz[:, j]).unsqueeze(dim=-1),
                                     z_fr)

        rootfr = rootfr1 - rootfr2
        l_aroot_layer_CC[:, :, j] = rootfr * l_aroot_tot

        v_aroot_layer[:, :, j]    = rootfr * 0.5 * (v_aroot_tot + v_troot)

    return v_ag, v_troot, l_aroot_layer_CC, v_aroot_layer
#=======================================================================================================================
def InitPlantKmax(  n_hypool_leaf,n_hypool_stem,
                    pl_sapwood,
                    kmax_node_pft,
                    p_taper,
                    rfrac_stem,
                    hydr_r2s,
                    l_aroot_layer_CC,
                    z_node_ag,
                    z_lower_ag,
                    z_upper_ag,
                    z_node_troot,
                    debug=True):

    nleavs          = n_hypool_leaf; nstems = n_hypool_stem
    nlevrhiz        = l_aroot_layer_CC.shape[-1]
    kmax_stem_upper = createTensor((*pl_sapwood.shape, nstems), fill_nan=False, dtype = fates_r8)
    kmax_stem_lower = torch.zeros_like(kmax_stem_upper)
    kmax_troot_lower= createTensor((*pl_sapwood.shape, nlevrhiz), fill_nan=False, dtype = fates_r8)


    min_pet_stem_dz = 0.00001

    # Assumed sapw_c = sapw_c_target

    # ! Force at least a small difference
    # ! in the top of stem and petiole

    # ! Stem Maximum Hydraulic Conductance

    for k in range(nstems):

        # ! index for "above-ground" arrays, that contain stem and leaf
        # ! in one vector
        k_ag = k + nleavs

        # ! Depth from the petiole to the lower, node and upper compartment edges

        z_lower = z_node_ag[:, :, nleavs - 1] - z_lower_ag[:, :, k_ag]
        z_node  = z_node_ag[:, :, nleavs - 1] - z_node_ag[:, :, k_ag]
        z_upper = torch.clamp(z_node_ag[:, :, nleavs - 1] - z_upper_ag[:, :, k_ag],
                              min=min_pet_stem_dz)

        # ! Then we calculate the maximum conductance from each the lower, node and upper
        # ! edges of the compartment to the petiole. The xylem taper factor requires
        # ! that the kmax it is scaling is from the point of interest to the mean height
        # ! of the petioles.  Then we can back out the conductance over just the path
        # ! of the upper and lower compartments, but subtracting them as resistors in
        # ! series.

        # ! max conductance from upper edge to mean petiole height
        # ! If there is no height difference between the upper compartment edge and
        # ! the petiole, at least give it some nominal amount to void FPE's

        kmax_upper = kmax_node_pft * F.xylemtaper(p_taper, z_upper) * pl_sapwood / z_upper

        # ! max conductance from node to mean petiole height
        kmax_node = kmax_node_pft * F.xylemtaper(p_taper, z_node) * pl_sapwood / z_node

        # ! max conductance from lower edge to mean petiole height
        kmax_lower = kmax_node_pft * F.xylemtaper(p_taper, z_lower) * pl_sapwood / z_lower

        # ! Max conductance over the path of the upper side of the compartment
        kmax_stem_upper[:, :, k] = (1. / kmax_node - 1. / kmax_upper) ** (-1.)

        # ! Max conductance over the path on the loewr side of the compartment
        kmax_stem_lower[:, :, k] = (1. / kmax_lower - 1. / kmax_node) ** (-1.)

        if (debug):
            # ! The following clauses should never be true:
            if (z_lower < z_node).any() or (z_node < z_upper).any():
                print('Problem calculating stem Kmax')
                # print(z_lower, z_node, z_upper)
                # print(kmax_lower * z_lower, kmax_node * z_node, kmax_upper * z_upper)
                sys.exit()

    # ! Maximum conductance of the upper compartment in the transporting root
    # ! that connects to the lowest stem (btw: z_lower_ag(n_hypool_ag) == 0)
    z_upper    = z_lower_ag[:, :, nleavs - 1]
    z_node     = z_lower_ag[:, :, nleavs - 1] - z_node_troot
    kmax_node  = kmax_node_pft * F.xylemtaper(p_taper, z_node) * pl_sapwood / z_node
    kmax_upper = kmax_node_pft * F.xylemtaper(p_taper, z_upper) * pl_sapwood / z_upper

    kmax_troot_upper = (1. / kmax_node - 1. / kmax_upper) ** (-1.)

    rmin_ag = 1. / kmax_petiole_to_leaf + \
              torch.sum(1. / kmax_stem_upper[:, :, : nstems], dim=-1) + \
              torch.sum(1. / kmax_stem_lower[:, :, : nstems], dim=-1) + \
              1. / kmax_troot_upper

    kmax_bg = 1. / (rmin_ag * (1. / rfrac_stem - 1.))

    kmax_aroot_upper      = torch.zeros_like(kmax_troot_lower)
    kmax_aroot_lower      = torch.zeros_like(kmax_troot_lower)
    kmax_aroot_radial_in  = torch.zeros_like(kmax_troot_lower)
    kmax_aroot_radial_out = torch.zeros_like(kmax_troot_lower)
    # ! The max conductance of each layer is in parallel, therefore
    # ! the kmax terms of each layer, should sum to kmax_bg
    sum_l_aroot = torch.sum(l_aroot_layer_CC, dim=-1)
    for j in range(nlevrhiz):
        kmax_layer = kmax_bg * l_aroot_layer_CC[:, :, j] / sum_l_aroot

        # ! Surface area of the absorbing roots for a single plant in this layer [m2]
        surfarea_aroot_layer = 2. * pi_const * hydr_r2s * l_aroot_layer_CC[:, :, j]

        kmax_troot_lower[:, :, j] = 3.0 * kmax_layer

        kmax_aroot_upper[:, :, j] = 3.0 * kmax_layer

        kmax_aroot_lower[:, :, j] = 3.0 * kmax_layer

        # ! Convert from surface conductivity [kg H2O m-2 s-1 MPa-1] to [kg H2O s-1 MPa-1]
        kmax_aroot_radial_in[:, :, j] = hydr_kmax_rsurf1 * surfarea_aroot_layer

        kmax_aroot_radial_out[:, :, j] = hydr_kmax_rsurf2 * surfarea_aroot_layer

    return kmax_stem_upper, kmax_stem_lower, kmax_troot_upper, kmax_troot_lower, kmax_aroot_upper, kmax_aroot_lower, kmax_aroot_radial_in, kmax_aroot_radial_out
#=======================================================================================================================
def InitSizeDepRhizVolLenCon(nshell,
                             si_code,
                             l_aroot_layer_CC,
                             st_density,
                             dz,
                             hksat_sisl,dz_sisl,
                             map_r2s,
                             mean_type):
    nlevrhiz            = l_aroot_layer_CC.shape[-1]
    nsites              = l_aroot_layer_CC.shape[0]

    l_aroot_layer_init  = createTensor((nsites, nlevrhiz), fill_nan=True, dtype=fates_r8)
    l_aroot_layer       = createTensor((nsites, nlevrhiz), fill_nan=True, dtype=fates_r8)

    r_out_shell         = createTensor((nsites, nlevrhiz, nshell), fill_nan=True, dtype = fates_r8)
    r_node_shell        = createTensor((nsites, nlevrhiz, nshell), fill_nan=True, dtype = fates_r8)
    v_shell             = createTensor((nsites, nlevrhiz, nshell), fill_nan=True, dtype = fates_r8)
    kmax_upper_shell    = createTensor((nsites, nlevrhiz, nshell), fill_nan=True, dtype = fates_r8)
    kmax_lower_shell    = createTensor((nsites, nlevrhiz, nshell), fill_nan=True, dtype = fates_r8)
    # subroutine UpdateSizeDepRhizVolLenCon(currentSite, bc_in)

    # !
    # ! !DESCRIPTION: Updates size of 'representative' rhizosphere -- node radii, volumes of the site.
    # ! As fine root biomass (and thus absorbing root length) increases, this characteristic
    # ! rhizosphere shrinks even though the total volume of soil tapped by fine roots remains
    # ! the same.
    # !
    # ! !USES:
    #
    #
    # ! !ARGUMENTS:
    # type(ed_site_type)     , intent(inout), target :: currentSite
    # type(bc_in_type)       , intent(in) :: bc_in
    #
    # !
    # ! !LOCAL VARIABLES:
    # type(ed_site_hydr_type), pointer :: csite_hydr
    # type(fates_patch_type)  , pointer :: cPatch
    # type(fates_cohort_type) , pointer :: cCohort
    # type(ed_cohort_hydr_type), pointer :: ccohort_hydr
    # real(r8)                       :: hksat_s                      ! hksat converted to units of 10^6sec
    #                                                              ! which is equiv to       [kg m-1 s-1 MPa-1]
    # integer                        :: j,k                          ! gridcell, soil layer, rhizosphere shell indices
    # integer                        :: j_bc                         ! soil layer index of boundary condition
    # real(r8)                       :: large_kmax_bound = 1.e4_r8   ! for replacing kmax_bound_shell wherever the
    #                                                              ! innermost shell radius is less than the assumed
    #                                                              ! absorbing root radius rs1
    #                                                              ! 1.e-5_r8 from Rudinger et al 1994
    # integer                        :: nlevrhiz

    # ! Note, here is where the site level soil depth/layer is set
    # ! update cohort-level root length density and accumulate it across cohorts and patches to the column level
    nshell   = r_out_shell.shape[-1]
    l_aroot_layer[:]  = 0.0

    #************************************************#
    l_aroot_layer_CC = l_aroot_layer_CC.mean(dim=1)
    # Convert the codes to integer labels
    unique_codes, codes_labels = np.unique(si_code, return_inverse=True)
    # Convert codes_labels to a tensor
    codes_labels = torch.tensor(codes_labels)
    # Calculate the mean along the first dimension based on the labels
    result = torch.stack([l_aroot_layer_CC[codes_labels == i].mean(dim=0) for i in range(len(unique_codes))])
    # Map the result back to match the order of the original codes
    l_aroot_layer_CC = result[codes_labels]
    l_aroot_layer = l_aroot_layer + l_aroot_layer_CC * st_density
    # ************************************************#

    # l_aroot_layer = l_aroot_layer + l_aroot_layer_CC.mean(dim=1) * st_density

    # ! update outer radii of column-level rhizosphere shells (same across patches and cohorts)
    # ! Provisions are made inside shellGeom() for layers with no roots

    r_out_shell, r_node_shell, v_shell  = F.shellGeom(l_aroot_layer,
                                                      dz,
                                                      r_out_shell,
                                                      r_node_shell,
                                                      v_shell)

    hksat_s = F.AggBCToRhiz(map_r2s,
                            hksat_sisl,
                            dz_sisl,
                            mean_type) * m_per_mm * 1. / grav_earth * pa_per_mpa

    mask1 = l_aroot_layer != l_aroot_layer_init
    mask2 = l_aroot_layer > nearzero
    mask12= (mask1 * mask2).bool()

    for k in range(nshell):
        if k==0:
            # ! Set the max conductance on the inner shell first.  If the node radius
            # ! on the shell is smaller than the root radius, just set the max conductance
            # ! to something extremely high.
            mask3 = r_node_shell[:, :, k] <= rs1
            mask = mask12 * mask3
            mask = mask.bool()

            kmax_upper_shell[:, :, k] = torch.where(mask,
                                                    large_kmax_bound,
                                                    2. * pi_const * l_aroot_layer / torch.log(r_node_shell[:, :, k] / rs1) * hksat_s)

        else:
            kmax_upper_shell[:,:,k] = torch.where(mask12,
                                                  2.0 * pi_const * l_aroot_layer /torch.log(r_node_shell[:,:,k]/r_out_shell[:, :,k-1]) * hksat_s,
                                                  kmax_upper_shell[:,:,k])

        kmax_lower_shell[:,:,k] = torch.where(mask12,
                                              2. * pi_const * l_aroot_layer / torch.log(r_out_shell[:,:,k]/r_node_shell[:,:,k])*hksat_s,
                                              kmax_lower_shell[:,:,k])

    return l_aroot_layer, r_out_shell, r_node_shell, v_shell, kmax_upper_shell, kmax_lower_shell
#=======================================================================================================================
def InitPlantHydStates(init_mode,
                       h2osoi_liqvol_shell,
                       l_aroot_layer_CC,
                       z_node_troot,
                       z_node_ag,
                       zi_rhiz,
                       dz_rhiz,
                       wrf_soil, wrf_plant, wkf_plant):


    wkf_stomata = wkf_plant['stomata_p_media']
    n_hypool_ag = z_node_ag.shape[-1]
    nlevrhiz    = l_aroot_layer_CC.shape[-1]
    psi_aroot   = torch.zeros_like(l_aroot_layer_CC)
    psi_ag      = torch.zeros_like(z_node_ag)
    # ===============================================================================
    # ===============================================================================
    if init_mode == 2:
        mask = (l_aroot_layer_CC > nearzero).bool()
        #  ! Match the potential of the absorbing root to the inner rhizosphere shell
        psi_aroot[:] = torch.where(mask,
                                  wrf_soil.psi_from_th(h2osoi_liqvol_shell[:,:,0].unsqueeze(dim = 1)),
                                  psi_aroot_init)
    else:
        psi_aroot[:] = psi_aroot_init

    h_aroot_mean,_ = torch.min(psi_aroot + mpa_per_pa  * denh2o * grav_earth *
                               (-zi_rhiz +0.5 * dz_rhiz).unsqueeze(dim=1), dim = -1)
    # ! initialize plant water potentials with slight potential gradient (or zero) (dh/dz = C)
    # ! the assumption is made here that initial conditions for soil water will
    # ! be in (or at least close to) hydrostatic equilibrium as well, so that
    # ! it doesn't matter which absorbing root layer the transporting root water
    #
    #
    # ! Set the transporting root to be in equilibrium with mean potential
    # ! of the absorbing roots, minus any gradient we add
    # ===============================================================================
    # ===============================================================================
    psi_troot = h_aroot_mean - mpa_per_pa * denh2o * grav_earth * z_node_troot - dh_dz
    # ===============================================================================
    # ===============================================================================
    # ! working our way up a tree, assigning water potentials that are in
    # ! hydrostatic equilibrium (minus dh_dz offset) with the water potential immediately below
    dz              = z_node_ag[:,:, -1] - z_node_troot
    psi_ag[:,:, -1] = psi_troot - mpa_per_pa * denh2o * grav_earth * dz - dh_dz
    for k in range(n_hypool_ag - 2, -1, -1):
        dz            = z_node_ag[:,:,k]  - z_node_ag[:,:,k + 1]
        psi_ag[:,:,k] = psi_ag[:,:,k + 1] - mpa_per_pa * denh2o * grav_earth * dz - dh_dz
    # ===============================================================================
    # ===============================================================================
    psi_plant= torch.cat([psi_ag, psi_troot.unsqueeze(dim=-1), psi_aroot], dim = -1) # troot =1 , therefore need unsqueeze
    th_plant = torch.maximum(wrf_plant.th_from_psi(psi_plant), wrf_plant.get_thmin())
    ftc_plant= wkf_plant['plant_media'].ftc_from_psi(psi_plant)

    if init_mode == 2:
        th_plant[:,:,-nlevrhiz:][~mask] = 0.0
    # ===============================================================================
    # ===============================================================================
    # ! leaf water potential limitation on gs
    btran = wkf_stomata.ftc_from_psi(psi_ag[:,:,0])
    # ===============================================================================
    # ===============================================================================
    # Check for positive pressures
    if  (psi_troot > 0.0).any() or (psi_ag > 0.0).any() or (psi_aroot > 0.0).any():
        print('Initialized plant compartments with positive pressure?')
        print('psi troot:', psi_troot[torch.where(psi_troot > 0.0)])
        print('psi ag:'   , psi_ag[torch.where(psi_ag > 0.0)])
        print('psi_aroot:', psi_aroot[torch.where(psi_aroot > 0.0)])
        # Terminate the program with an error message
        sys.exit('Error: Initialized plant compartments with positive pressure')

    return th_plant, btran, psi_plant, ftc_plant
#=======================================================================================================================
def UpdateSoilHydStates(  dz_sisl,
                        eff_porosity_sl,
                        h2o_liq_sisl,
                        dz_rhiz,
                        map_r2s,
                        nshell,
                        mean_type):
    h2osoi_liqvol_shell = createTensor((*dz_sisl.shape, nshell), fill_nan=False, dtype = fates_r8)

    eff_por = F.AggBCToRhiz(map_r2s, eff_porosity_sl,dz_sisl, mean_type)

        # ! [kg/m2] / ([m] * [kg/m3]) = [m3/m3]
    h2osoi_liqvol = torch.minimum(eff_por, torch.sum(h2o_liq_sisl.unsqueeze(dim=-1) * map_r2s.unsqueeze(0), dim = -1)/(dz_rhiz*denh2o))
    h2osoi_liqvol_shell[:, :,:nshell] = h2osoi_liqvol.unsqueeze(dim = -1)
    return h2osoi_liqvol_shell

#=======================================================================================================================
def setRhizlayers(nlevsoil,nlevrhiz,zi_sisl, dz_sisl, aggmeth):
    nsites     = zi_sisl.shape[0]

    map_r2s    = createTensor((nlevrhiz, 2), fill_nan=False, dtype=fates_int, value= -999)
    zi_rhiz    = createTensor((nsites, nlevrhiz), fill_nan=True, dtype=fates_r8)
    dz_rhiz    = createTensor((nsites, nlevrhiz), fill_nan=True, dtype=fates_r8)

    if aggmeth == "rhizlayer_aggmeth_none":
        for j in range(nlevrhiz):
            map_r2s[j, 0] = j
            map_r2s[j, 1] = j
            zi_rhiz[:, j] = zi_sisl[:, j]
            dz_rhiz[:, j] = dz_sisl[:, j]

    elif aggmeth == "rhizlayer_aggmeth_combine12":
        map_r2s[0, 0] = 0
        j_bc = min(1, nlevsoil)
        map_r2s[0, 1] = j_bc
        zi_rhiz[:, 0] = zi_sisl[:, j_bc]
        dz_rhiz[:, 0] = dz_sisl[:, :j_bc + 1].sum(dim=1)

        for j in range(1, nlevrhiz):
            map_r2s[j, 0] = j + 1
            map_r2s[j, 1] = j + 1
            zi_rhiz[:, j] = zi_sisl[:, j + 1]
            dz_rhiz[:, j] = dz_sisl[:, j + 1]

    elif aggmeth == "rhizlayer_aggmeth_balN":
        ntoagg = int(np.ceil(float(nlevsoil) / float(nlevrhiz) - nearzero))

        if (ntoagg < 1):
            raise ValueError(
                f'rhizosphere balancing method rhizlayer_aggmeth_balN is failing to get a starting estimate of soil layers per rhiz layers: {ntoagg}')

        # ! This array defines the number of soil layers
        # ! in each rhiz layer, start off with a max value
        # ! then we incrementally work our way from bottom up
        # ! reducing this number, until the number of soil
        # ! layers in the array matches the total actual

        ns_per_rhiz = np.full((nlevrhiz,), ntoagg)
        # Define the loop to adjust ns_per_rhiz
        while sum(ns_per_rhiz) > nlevsoil:
            for j in range(len(ns_per_rhiz) - 1, -1, -1):  # Iterate backward through ns_per_rhiz
                ns_per_rhiz[j] -= 1

                if sum(ns_per_rhiz) <= nlevsoil:
                    break  # Exit the inner loop if the condition is met

                if ns_per_rhiz[j] == 0:
                    # Log an error message and exit
                    raise ValueError('rhizosphere balancing method rhizlayer_aggmeth_balN produced a '
                                     'rhizosphere layer with 0 soil layers...exiting')

        # ! Assign the mapping
        map_r2s[0, 0] = 0
        for j in range(nlevrhiz - 1):
            j_t = map_r2s[j, 0].unique().item()
            j_b = j_t + ns_per_rhiz[j].item() - 1
            map_r2s[j, 1] = j_b
            map_r2s[j + 1, 0] = j_b + 1
            zi_rhiz[:, j] = zi_sisl[:, j_b]
            dz_rhiz[:, j] = torch.sum(dz_sisl[:, j_t:j_b + 1], dim=1)

        j_t = map_r2s[-1, 0].unique().item()
        j_b = j_t + ns_per_rhiz[-1] - 1
        map_r2s[-1, 1] = j_b
        zi_rhiz[:, -1] = zi_sisl[:, j_b]
        dz_rhiz[:, -1] = torch.sum(dz_sisl[:, j_t:j_b + 1], dim=1)

    else:
        raise RuntimeError('You specified an undefined rhizosphere layer aggregation method')

    # Determine the size of the matrix
    num_rows = map_r2s.shape[0]
    num_cols = num_rows  # Assuming a square matrix, adjust as necessary

    # Create an empty dense matrix first (you can convert it to sparse later if needed)
    dense_matrix = torch.zeros(num_rows, num_cols).to(map_r2s.device)

    # Fill the dense matrix according to the index ranges
    for row_idx, (start, end) in enumerate(map_r2s):
        dense_matrix[row_idx, start:end + 1] = 1

    # Optionally convert the dense matrix to a sparse format
    map_r2s_sparse = dense_matrix  # .to_sparse()
    return map_r2s, map_r2s_sparse, zi_rhiz, dz_rhiz
#=======================================================================================================================
def SetConnections(n_hypool_leaf, n_hypool_stem,
                   n_hypool_troot,n_hypool_aroot ,
                   nlevrhiz, nshell,
                   num_nodes,num_connections,
                   hydro_media_id,
                   ):
    node_layer = createTensor(num_nodes, dtype=torch.long)
    conn_up    = createTensor(num_connections, dtype=torch.long)
    conn_dn    = createTensor(num_connections, dtype=torch.long)
    pm_node    = createTensor(num_nodes, dtype=torch.long)
    icnx_      = createTensor(nlevrhiz, dtype=torch.long)
    inode_     = createTensor(nlevrhiz, dtype=torch.long)
    # self routine should be updated
    # when new layers are added as plants grow into them?
    # **************************************************************************************************************
    n_hypool_ag   = n_hypool_leaf + n_hypool_stem
    num_cnxs = 0
    num_nds  = 0

    for k in range(1, n_hypool_leaf + 1):
        num_cnxs += 1
        num_nds += 1
        conn_dn[num_cnxs - 1] = k  # Leaf is the dn, origin, bottom
        conn_up[num_cnxs - 1] = k + 1
        pm_node[num_nds - 1] = hydro_media_id['leaf_p_media']

    for k in range(n_hypool_leaf + 1, n_hypool_ag + 1):
        num_cnxs += 1
        num_nds += 1
        conn_dn[num_cnxs - 1] = k
        conn_up[num_cnxs - 1] = k + 1
        pm_node[num_nds - 1] = hydro_media_id['stem_p_media']

    num_nds     = n_hypool_ag + n_hypool_troot
    node_tr_end = num_nds
    nt_ab       = n_hypool_ag + n_hypool_troot + n_hypool_aroot
    num_cnxs    = n_hypool_ag

    pm_node[num_nds - 1]       = hydro_media_id['troot_p_media']
    node_layer[0:n_hypool_ag]  = 0
    node_layer[num_nds - 1]    = 1

    for j in range(1,nlevrhiz + 1):
        for k in range(1, n_hypool_aroot +nshell + 1):
            num_nds     += 1
            num_cnxs    += 1
            node_layer[num_nds - 1] = j
            if k == 1:  # Troot-aroot
                # Junction node
                conn_dn[ num_cnxs - 1] = node_tr_end  # Absorbing root
                conn_up[ num_cnxs - 1] = num_nds
                pm_node[ num_nds - 1] = hydro_media_id['aroot_p_media']
            else:
                conn_dn[ num_cnxs - 1] = num_nds - 1
                conn_up[ num_cnxs - 1] = num_nds
                pm_node[ num_nds - 1] = hydro_media_id['rhiz_p_media']
    # **************************************************************************************************************
    # Add two index tensors for updatekmaxdn
    icnx_init = (n_hypool_stem - 1) + 1
    inode_init = (n_hypool_ag - 1)

    for j in range(nlevrhiz):
        icnx_[j] = icnx_init + (n_hypool_aroot +nshell) * (j + 1)
        inode_[j] = inode_init + (n_hypool_aroot +nshell) * (j + 1)
    # **************************************************************************************************************
    mask_leaf_media = pm_node == hydro_media_id['leaf_p_media']
    mask_stem_media = pm_node == hydro_media_id['stem_p_media']
    mask_troot_media = pm_node == hydro_media_id['troot_p_media']
    mask_aroot_media = pm_node == hydro_media_id['aroot_p_media']
    mask_plant_media = mask_leaf_media + mask_stem_media + mask_troot_media + mask_aroot_media
    # **************************************************************************************************************
    return conn_dn, conn_up, pm_node,node_layer, icnx_, inode_, mask_plant_media
#=======================================================================================================================
def InitPlantWTF(plant_wrf_type,
                 n_hypool_plant,
                 n_hypool_aroot,
                 nlevrhiz,
                 numpft,
                 vg_Dict, tfs_wrfDict, tfs_wkfDict, tfstomata_Dict,
                 hydro_media_name, hydro_media_id,
                 rwccap, rwcft,
                 sites_pft,
                 pm_node,
                 ):

    #=====================
    dims          = sites_pft.shape
    select_case   = plant_wrf_type
    n_plant_nodes = n_hypool_plant

    # Initialize dictionaries
    #========================
    wrf_plant = {}
    wkf_plant = {'plant_media':[], 'stomata_p_media':[]}


    if select_case == "van_genuchten_type":
        wrf_vg = wrf_type_vg()
        wkf_vg = wkf_type_vg()
        # For this media van_genuchten_type is selected
        params_in = {param: createTensor(dims=dims + (n_plant_nodes,), dtype=fates_r8, value=fates_unset_r8) for param in
                     vg_Dict.keys()}
        for media in hydro_media_name:
            if media == 'aroot_p_media':
                mask_media  = createTensor(n_plant_nodes, torch.bool)
                aroot_index = n_hypool_aroot * nlevrhiz
                mask_media[-aroot_index:] = True
            else:
                mask_media = (pm_node ==  hydro_media_id[media])[:n_plant_nodes]

            mask_media = mask_media.unsqueeze(0).unsqueeze(0)
            for ft_id in range(numpft):
                mask = (sites_pft == ft_id).unsqueeze(dim=-1)
                mask = mask * mask_media
                #================================================
                for key, val in vg_Dict.items():
                    try:
                        params_in[key] = torch.where(mask, val[media][ft_id], params_in[key])
                    except:  # plant_tort
                        params_in[key] = torch.where(mask, val, params_in[key])

        # Initialize the class instances
        wrf_plant = wrf_vg
        wrf_plant.set_wrf_param(list(params_in.values())[:-1])

        wkf_plant['plant_media'] = wkf_vg
        wkf_plant['plant_media'].set_wkf_param(list(params_in.values()))

    elif select_case == "tfs_type":
        # For this media tfs type is selected

        wrf_tfs = wrf_type_tfs()
        wkf_tfs = wkf_type_tfs()

        params_wrf = {param: createTensor(dims=dims + (n_plant_nodes,), dtype=fates_r8, value=fates_unset_r8) for param in
                      tfs_wrfDict.keys()}
        params_wkf = {param: createTensor(dims=dims + (n_plant_nodes,), dtype=fates_r8, value=fates_unset_r8) for param in
                      tfs_wkfDict.keys()}

        #====================================================
        for media in hydro_media_name:
            if media == 'aroot_p_media':
                mask_media  = createTensor(n_plant_nodes, torch.bool)
                aroot_index = n_hypool_aroot * nlevrhiz
                mask_media[-aroot_index:] = True
            else:
                mask_media = (pm_node ==  hydro_media_id[media])[:n_plant_nodes]
            mask_media = mask_media.unsqueeze(0).unsqueeze(0)
            #===============================================================================================
            if media == "leaf_p_media":
                cap_slp = 0.0
                cap_int = 0.0
                cap_corr = 1.0
            else:
                cap_slp = (fates_hydro_psi0 - fates_hydro_psicap) / (1.0 - rwccap[media])
                cap_int = -cap_slp + fates_hydro_psi0
                cap_corr = -cap_int / cap_slp

            for ft_id in range(numpft):
                mask = (sites_pft == ft_id).unsqueeze(dim=-1)
                mask = mask * mask_media
                #================================================
                for key, val in tfs_wrfDict.items():
                    try:
                        params_wrf[key] = torch.where(mask, val[media][ft_id], params_wrf[key])
                    except:
                        if key == 'cap_slp':
                            params_wrf[key] = torch.where(mask, cap_slp, params_wrf[key])
                        elif key == 'cap_int':
                            params_wrf[key] = torch.where(mask, cap_int, params_wrf[key])
                        elif key == 'cap_corr':
                            params_wrf[key] = torch.where(mask, cap_corr, params_wrf[key])
                        elif key == "rwc_ft":
                            params_wrf[key] = torch.where(mask, rwcft[media], params_wrf[key])
                        elif key == "pmedia":
                            params_wrf[key] = torch.where(mask, hydro_media_id[media], params_wrf[key])

                for key, val in tfs_wkfDict.items():
                    params_wkf[key] = torch.where(mask, val[media][ft_id], params_wkf[key])

        params_wrf['pmedia'] = params_wrf['pmedia'].to(torch.int64)

        wrf_plant = wrf_tfs
        wrf_plant.set_wrf_param(list(params_wrf.values()))
        wkf_plant['plant_media'] = wkf_tfs
        wkf_plant['plant_media'].set_wkf_param(list(params_wkf.values()))

    # for stomata_p_media
    wkf_tfs = wkf_type_tfs()
    params_wkf = {param: createTensor(dims=dims, dtype=fates_r8, value=fates_unset_r8) for param in
                  tfstomata_Dict.keys()}

    for ft_id in range(numpft):
        mask = sites_pft == ft_id
        for key, val in tfstomata_Dict.items():
            params_wkf[key] = torch.where(mask, val[ft_id], params_wkf[key])

    wkf_plant['stomata_p_media'] = wkf_tfs
    wkf_plant['stomata_p_media'].set_wkf_param(list(params_wkf.values()))

    return wrf_plant, wkf_plant

#=======================================================================================================================
def InitSoilWTF( soil_wrf_type, soil_wkf_type, mean_type,
                 nlevrhiz, numpft,
                 vg_Dict,
                 dz_sisl,
                 watsat_sisl,
                 sucsat_sisl,
                 bsw_sisl,
                 sites_pft,
                 map_r2s
                 ):
    # HydrSiteColdStart

    dims            = sites_pft.shape
    params_in = {param: createTensor(dims=dims, dtype=fates_r8, value=fates_unset_r8) for param
                 in
                 vg_Dict.keys()}

    for ft_id in range(numpft):
        mask = sites_pft == ft_id
        for key, val in vg_Dict.items():
            params_in[key] = torch.where(mask, val, params_in[key])

    watsat, sucsat, bsw = [
        F.AggBCToRhiz(map_r2s, param, dz_sisl, mean_type).unsqueeze(dim=1).expand(dims + (-1,))
        for param in [watsat_sisl, sucsat_sisl, bsw_sisl]
    ]

    if soil_wrf_type == "van_genuchten_type":
        wrf_vg = wrf_type_vg()
        wrf_soil = wrf_vg
        wrf_soil.set_wrf_param([tensor.unsqueeze(dim=-1).expand(dims + (nlevrhiz,))
                                for tensor in list(params_in.values())[:-1]])


    elif soil_wrf_type == "campbell_type":
        wrf_cch = wrf_type_cch()
        wrf_soil = wrf_cch
        wrf_soil.set_wrf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw])

    elif soil_wrf_type == wrf_type_smooth_cch:
        wrf_smooth_cch = wrf_type_smooth_cch()
        wrf_soil = wrf_smooth_cch
        wrf_soil.set_wrf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw, 1.0])

    elif soil_wrf_type == "smooth2_campbell_type":
        wrf_smooth_cch = wrf_type_smooth_cch()
        wrf_soil = wrf_smooth_cch
        wrf_soil.set_wrf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw, 2.0])

    elif soil_wrf_type == "tfs_type":
        raise RuntimeError('TFS water retention curves not available for soil')

    else:
        raise RuntimeError('undefined water retention type for soil:', soil_wrf_type)

    if soil_wkf_type == "van_genuchten_type":

        wkf_vg = wkf_type_vg()
        wkf_soil = wkf_vg
        wkf_soil.set_wkf_param([tensor.unsqueeze(dim=-1).expand(dims + (nlevrhiz,))
                                for tensor in list(params_in.values())])

    elif soil_wkf_type == "campbell_type":

        wkf_cch = wkf_type_cch()
        wkf_soil = wkf_cch
        wkf_soil.set_wkf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw])

    elif soil_wkf_type == "smooth1_campbell_type":
        wkf_cch = wkf_type_smooth_cch()
        wkf_soil = wkf_cch
        wkf_soil.set_wkf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw, 1.])

    elif soil_wkf_type == "smooth2_campbell_type":
        wkf_cch = wkf_type_smooth_cch()
        wkf_soil = wkf_cch
        wkf_soil.set_wkf_param([watsat, (-1.0) * sucsat * denh2o * grav_earth * mpa_per_pa * m_per_mm, bsw, 2.])

    elif soil_wkf_type == "tfs_type":
        raise RuntimeError('TFS conductance not used in soil')

    else:
        raise RuntimeError('undefined water conductance type for soil:', soil_wkf_type)

    return wrf_soil, wkf_soil

#=======================================================================================================================