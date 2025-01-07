import numpy as np
from conf.Constants import *
from conf.Constants import dens_fresh_liquid_water as denh2o, t_water_freeze_k_1atm as tfrz, area as area_site
from data.utils import createTensor
from physical.biogeochem.AllometryMod import decay_coeff_vcmax
#========================================================================================================
def AggBCToRhiz(map_r2s, var_in, weight, mean_type):
    # var_out = torch.zeros_like(var_in[:,j]).clone()
    # I completely changed the indexing here so please be careful when initializing map_r2s
    var_in = var_in.unsqueeze(-1) * map_r2s.unsqueeze(0)
    weight = weight.unsqueeze(-1) * map_r2s.unsqueeze(0)

    if mean_type == "arithmetic_mean":
        var_out = torch.sum(var_in * weight, dim=-1) / torch.sum(weight,dim=-1)
    else:
        var_out = torch.sum(weight, dim=-1) / torch.nansum(weight / var_in,dim=-1)

    return var_out
#========================================================================================================
def zeng2001_crootfr(a, b, z, z_max=None, debug = False):
    """
    Calculate the cumulative root fraction for tensor inputs.

    Arguments:
    a, b  -- pft parameters, tensors or floats
    z     -- soil depth (m), tensor
    z_max -- optional, max soil depth (m), tensor or float

    Returns:
    crootfr -- cumulative root fraction, tensor
    """
    crootfr = 1.0 - 0.5 * (torch.exp(-a * z) + torch.exp(-b * z))

    if z_max is not None:

        # Adjust crootfr if z_max is provided
        min_z_zmax  = torch.minimum(z, z_max)
        crootfr     = 1.0 - 0.5 * (torch.exp(-a * min_z_zmax) + torch.exp(-b * min_z_zmax))
        crootfr_max = 1.0 - 0.5 * (torch.exp(-a * z_max) + torch.exp(-b * z_max))
        crootfr    /= crootfr_max
        if debug:
            # Example debug information, adapt as necessary for your logging system
            if torch.any(crootfr_max < nearzero) or torch.any(crootfr_max > 1):
                print('Problem scaling crootfr in zeng2001_crootfr_tensor')
                print('z_max:', z_max)
                print('crootfr_max:', crootfr_max)

    return crootfr
#========================================================================================================
def bisect_rootfr(a, b, z_max, lower, upper, xtol, ytol, crootfr, max_iter = 100):
    # !
    # ! !DESCRIPTION: Bisection routine for getting the inverse of the cumulative root
    # !  distribution. No analytical soln bc crootfr ~ exp(ax) + exp(bx).

    f_lo  = zeng2001_crootfr(a, b, lower, z_max) - crootfr
    f_hi  = zeng2001_crootfr(a, b, upper, z_max) - crootfr
    chg   = upper - lower

    mid  = 0.5 * (lower + upper)  # should be the same

    if (torch.abs(chg) <= xtol).all():
        print("Cannot enter solver iterations since upper bound = lower bound")
        x_new = mid
    else:
        x_new = torch.zeros_like(mid)
        for nitr in range(max_iter):

            mid   = 0.5*(lower + upper)
            f_mid = zeng2001_crootfr(a, b, mid, z_max) - crootfr

            # Update bounds and function values
            lower_update = (f_lo * f_mid) < 0
            upper_update = (f_hi * f_mid) < 0

            lower[upper_update] = mid[upper_update]
            upper[lower_update] = mid[lower_update]
            chg = upper - lower

            # Check for convergence
            converged = (torch.abs(f_mid) <= ytol) | (torch.abs(chg) <= xtol)
            # I moved this line here to make sure it updates x_new. This is the only difference from Fortran version
            # ===============================
            x_new[converged] = mid[converged]
            if converged.all():
                break
            # x_new[converged] = mid[converged]
            if (nitr == max_iter - 1):
                print('Warning: number of iteraction exceeds maximum iterations for bisect_rootfr')
                print("f(x) = ", torch.abs(f_mid).max())
                x_new = mid
    # import numpy as np
    # np.set_printoptions(suppress=True, precision=6)
    # print(x_new.numpy())
    return x_new
#========================================================================================================
def GetKAndDKDPsi(kmax_dn,kmax_up, h_dn,h_up, ftc_dn,ftc_up, do_upstream_k):

    # ! -----------------------------------------------------------------------------
    # ! This routine will return the effective conductance "K", as well
    # ! as two terms needed to calculate the implicit solution (using taylor
    # ! first order expansion).  The two terms are generically named A & B.
    # ! Thus the name "KAB".  These quantities are specific not to the nodes
    # ! themselves, but to the path between the nodes, defined as positive
    # ! direction from "up"per (closer to atm) and "lo"wer (further from atm).
    # ! -----------------------------------------------------------------------------


    #
    # ! We use the local copies of the FTC in our calculations
    # ! because we don't want to over-write the global values.  This prevents
    # ! us from overwriting FTC on nodes that have more than one connection

    ftc_dnx = ftc_dn
    ftc_upx = ftc_up


    # ! Calculate difference in total potential over the path [MPa]

    h_diff  = h_up - h_dn

    # ! If we do enable "upstream K", then we are saying that
    # ! the fractional loss of conductivity is dictated
    # ! by the upstream side of the flow.  In this case,
    # ! the change in ftc is only non-zero on that side, and is
    # ! zero'd otherwise.

    if(do_upstream_k):

        mask = (h_diff > 0.)
        mask = mask.bool()
        ftc_dnx = torch.where(mask, ftc_up, ftc_dnx)
        ftc_upx = torch.where(~mask, ftc_dn, ftc_upx)

    # ! Calculate total effective conductance over path  [kg s-1 MPa-1]
    k_eff = 1./(1./(ftc_upx * kmax_up) + 1./(ftc_dnx * kmax_dn))
    return k_eff
#============================================================================================================
def xylemtaper(pexp, dz):

    # use FatesConstantsMod, only : pi => pi_const

    # ! !DESCRIPTION: Following the theory presented i
    # ! Savage VM, Bentley LP, Enquist BJ, Sperry JS, Smith DD, Reich PB, von
    # ! Allmen EI. 2010.
    # ! Hydraulic trade-offs and space filling enable better predictions of
    # ! vascular structure
    # ! and function in plants. Proceedings of the National Academy of Sciences
    # ! 107(52): 22722-22727.

    # ! Revised 2019-01-03 BOC: total conductance exponent (qexp) is now a
    # ! continuous function of the xylem taper exponent (pexp).
    # ! renamed btap to qexp, a[tap][notap] to kN,
    # ! little_n to n_ext, to match variable names in Savage et al.

    lN   = 0.005
    n_ext= 2.

    a5 = -3.555547
    a4 =  9.760275
    a3 = -8.468005
    a2 =  1.096488
    a1 =  1.844792
    a0 =  1.320732

    qexp         = a5*pexp**5 + a4*pexp**4 + a3*pexp**3 + a2*pexp**2 + a1*pexp

    num          = 3.*torch.log(1. - dz/lN * (1.-n_ext**(1./3.)))
    den          = np.log(n_ext)
    big_N        = num/den - 1.
    r0rN         = n_ext**(big_N/2.)

    chi_tapnotap = r0rN**qexp

    return chi_tapnotap
#===========================================================================================================
def MaximumRootingDepth(sites_dbh, z_max_soil,
                        zroot_max_dbh,zroot_min_dbh,zroot_max_z,zroot_min_z,zroot_k,
                        ):

    # sites = ed_site_type(data)

    # ! ---------------------------------------------------------------------------------
    # ! Calculate the maximum rooting depth of the plant.
    # !
    # ! This is an exponential which is constrained by the maximum soil depth:
    # ! csite_hydr%zi_rhiz(nlevrhiz)
    # ! The dynamic root growth model by Junyan Ding, June 9, 2021
    # ! ---------------------------------------------------------------------------------
    #
    # real(r8),intent(in)  :: dbh               ! Plant dbh
    # integer,intent(in)   :: ft                ! Funtional type index
    # real(r8),intent(in)  :: z_max_soil        ! Maximum depth of soil (pos convention) [m]
    # real(r8),intent(out) :: z_fr              ! Maximum depth of plant's roots
    #                                         ! (pos convention) [m]
    #
    # real(r8) :: dbh_rel   ! Relative dbh of plant between the diameter at which we
    #                     ! define the shallowest rooting depth (dbh_0) and the diameter
    #                     ! at which we define the deepest rooting depth (dbh_max)
    #z_fr = createTensor(sites_pft.size(), value = fates_unset_r8, dtype = fates_r8)

    # for ft_id in range(numpft):
    #     mask    = sites_pft == ft_id
    #     dbh_max = zroot_max_dbh[ft_id]
    #     dbh_0   = zroot_min_dbh[ft_id]
    #     z_fr_max= zroot_max_z[ft_id]
    #     z_fr_0  = zroot_min_z[ft_id]
    #     frk     = zroot_k[ft_id]
    #
    #     dbh_rel = torch.clamp((torch.clamp(sites_dbh, min = dbh_0) - dbh_0) / (dbh_max - dbh_0),
    #                           max = 1.0)
    #
    #     z_fr = torch.where(mask,
    #                        z_fr_max/(1.0 + ((z_fr_max-z_fr_0)/z_fr_0) * torch.exp(-frk*dbh_rel)),
    #                        z_fr)
    dbh_max = zroot_max_dbh
    dbh_0   = zroot_min_dbh
    z_fr_max= zroot_max_z
    z_fr_0  = zroot_min_z
    frk     = zroot_k

    dbh_rel = torch.clamp((torch.clamp(sites_dbh, min = dbh_0) - dbh_0) / (dbh_max - dbh_0),
                          max = 1.0)

    z_fr    = z_fr_max/(1.0 + ((z_fr_max-z_fr_0)/z_fr_0) * torch.exp(-frk*dbh_rel))


    z_fr    = torch.minimum(z_fr, z_max_soil.unsqueeze(dim=-1))

    return z_fr
#===========================================================================================================
def shellGeom(l_aroot_layer,dz,r_out_shell,r_node_shell,v_shell):

    nshell       = r_out_shell.shape[-1]
    mask = l_aroot_layer <= nearzero  # of size nsites, nlevrhiz
    mask = mask.bool()
    #=====================================================================================================
    #  ! update outer radii of column-level rhizosphere shells (same across patches and cohorts)
    r_out_shell[:,:, -1]= (pi_const*l_aroot_layer/(area_site*dz))**(-0.5)
    if (nshell > 1) :
        for k in range(nshell-1):
            r_out_shell[:,:,k]   = rs1*(r_out_shell[:,:,-1]/rs1)**(k/nshell)  #! eqn(7) S98
    #=======================================================================================================
    r_out_shell = torch.where(mask.unsqueeze(dim=-1), 0.0, r_out_shell)

    for k in range(nshell):
        # Here I assume dist_bet_shells has the size of nsites, runlevel, nshell.
        # for each layer(row), each column represents the distance between shell1 & shell2, shell2 & shell3,....
        # r_out_shell[:,:,k] = "should be userdefined"
        if k == 0:
            r_node_shell[:, :, k] = torch.where(mask, 0.0, 0.5 * (rs1 + r_out_shell[:, :, k]))

            v_shell[:, :, k]      = torch.where(mask,
                                                0.0,
                                                pi_const * l_aroot_layer * (r_out_shell[:,:, k] ** 2. - rs1 ** 2.))
        else:

            r_node_shell[:, :, k] = torch.where(mask,
                                                0.0,
                                                0.5 * (r_out_shell[:, :, k - 1] + r_out_shell[:, :, k]))

            v_shell[:, :, k]      = torch.where(mask,
                                                0.0,
                                                pi_const * l_aroot_layer * (r_out_shell[:, :, k] ** 2. - r_out_shell[:, :, k - 1] ** 2.))


    return r_out_shell, r_node_shell, v_shell

#============================================================================================================
def SetMaxCondConnections(dims,
                          l_aroot_layer_CC,l_aroot_layer,
                          kmax_stem_upper , kmax_stem_lower,
                          kmax_troot_upper, kmax_troot_lower,
                          kmax_aroot_upper,kmax_aroot_lower,
                          kmax_aroot_radial_in,
                          kmax_upper_shell,kmax_lower_shell,
                          ):
    #   ! -------------------------------------------------------------------------------
    #   ! This subroutine sets the maximum conductances
    #   ! on the downstream (towards atm) and upstream (towards
    #   ! soil) side of each connection. This scheme is somewhat complicated
    #   ! by the fact that the direction of flow at the root surface impacts
    #   ! which root surface radial conductance to use, which makes these calculation
    #   ! dependent on the updating potential in the system, and not just a function
    #   ! of plant geometry and material properties.
    #   ! -------------------------------------------------------------------------------

    #    ! Set leaf to stem connections (only 1 leaf layer
    #    ! this will break if we have multiple, as there would
    #    ! need to be assumptions about which compartment
    #    ! to connect the leaves to.

    nlevrhiz         = l_aroot_layer.shape[-1]
    n_hypool_stem    = dims['n_hypool_stem']
    n_hypool_ag      = dims['n_hypool_ag']
    n_hypool_aroot   = dims['n_hypool_aroot']
    nshell           = dims['nshell']
    num_connections  = dims['num_connections']
    nsites, ncohorts = kmax_troot_upper.shape
    kmax_dn = createTensor((nsites, ncohorts, num_connections), dtype = fates_r8, value=fates_unset_r8)
    kmax_up = createTensor((nsites, ncohorts, num_connections), dtype = fates_r8, value=fates_unset_r8)


    icnx  = 0
    kmax_dn[:,:,icnx] = kmax_petiole_to_leaf
    kmax_up[:,:,icnx] = kmax_stem_upper[:,:,0]

    #  ! Stem to stem connections
    for istem in range(n_hypool_stem-1):
        icnx =   + 1
        kmax_dn[:,:,icnx] = kmax_stem_lower[:,:,istem]
        kmax_up[:,:,icnx] = kmax_stem_upper[:,:,istem+1]

    #  ! Path is between lowest stem and transporting root
    icnx  = icnx + 1
    kmax_dn[:,:,icnx] = kmax_stem_lower[:,:,n_hypool_stem - 1]
    kmax_up[:,:,icnx] = kmax_troot_upper

    # ! Path is between the transporting root and the absorbing roots
    inode = n_hypool_ag - 1
    for j  in range(nlevrhiz):

        aroot_frac_plant = l_aroot_layer_CC[:,:,j]/l_aroot_layer[:,j].unsqueeze(dim =-1)

        for k in range(n_hypool_aroot + nshell):
            icnx  = icnx + 1
            inode = inode + 1
            # !troot-aroot
            if ( k == 0 ):
                kmax_dn[:,:, icnx] = kmax_troot_lower[:,:,j]
                kmax_up[:,:, icnx] = kmax_aroot_upper[:,:,j]

            # ! aroot-soil
            elif( k == 1):

            # ! Special case. Maximum conductance depends on the
            # ! potential gradient.
                # mask = (h_node[:,:,inode] < h_node[:,:, inode+1] ).bool()

                # Updated this with simpler equation since already updated each timestep based on hnode in another simpler function
                # kmax_dn[:,:,icnx] = torch.where(mask,
                #                                 1. / (1. / kmax_aroot_lower[:, :, j] +1. / kmax_aroot_radial_in[:, :, j]),
                #                                 1. / (1. / kmax_aroot_lower[:, :, j] +1. / kmax_aroot_radial_out[:, :, j])
                #                                 )
                kmax_dn[:, :, icnx] = 1. / (1. / kmax_aroot_lower[:, :, j] + 1. / kmax_aroot_radial_in[:, :,j])
                kmax_up[:,:,icnx]   = kmax_upper_shell[:, j,0].unsqueeze(dim = -1) * aroot_frac_plant

            # ! soil - soil
            else:
                kmax_dn[:,:,icnx] = kmax_lower_shell[:, j,k-2].unsqueeze(dim = -1) *aroot_frac_plant
                kmax_up[:,:,icnx] = kmax_upper_shell[:, j,k-1].unsqueeze(dim = -1) *aroot_frac_plant

    return kmax_dn, kmax_up
#=======================================================================================================
def UpdatekmaxDn(kmax_dn,
                 kmax_aroot_lower,
                 kmax_aroot_radial_in, kmax_aroot_radial_out,
                 h_node,
                 icnx_, inode_):
    #   ! -------------------------------------------------------------------------------
    #   ! This subroutine sets the maximum conductances
    #   ! on the downstream (towards atm) and upstream (towards
    #   ! soil) side of each connection. This scheme is somewhat complicated
    #   ! by the fact that the direction of flow at the root surface impacts
    #   ! which root surface radial conductance to use, which makes these calculation
    #   ! dependent on the updating potential in the system, and not just a function
    #   ! of plant geometry and material properties.
    #   ! -------------------------------------------------------------------------------

    #    ! Set leaf to stem connections (only 1 leaf layer
    #    ! this will break if we have multiple, as there would
    #    ! need to be assumptions about which compartment
    #    ! to connect the leaves to.

    # ! Special case. Maximum conductance depends on the
    # ! potential gradient.
    mask = (h_node[...,inode_] < h_node[..., inode_+1])
    mask = mask.type(torch.uint8)

    # Updated this with simpler equation since already updated each timestep based on hnode in another simpler function
    kmax_dn[...,icnx_] = mask     * ( 1. / (1. / kmax_aroot_lower +1. / kmax_aroot_radial_in)) +\
                         (1-mask) * (1. / (1. / kmax_aroot_lower +1. / kmax_aroot_radial_out))

    return kmax_dn
# #=======================================================================================================
def GetZVnode(  n_hypool_troot,n_hypool_aroot,num_nodes,
                z_node_ag       ,v_ag       ,
                z_node_troot    ,v_troot    ,
                v_aroot_layer   ,
                v_shell         ,
                l_aroot_layer   ,l_aroot_layer_CC,
                zi_rhiz         ,dz_rhiz ):

    n_hypool_ag    = v_ag.shape[-1]
    nlevrhiz       = zi_rhiz.shape[-1]
    nshell         = v_shell.shape[-1]
    z_node         = createTensor((*v_troot.shape, num_nodes ), fill_nan=False, dtype = fates_r8)
    v_node         = createTensor((*v_troot.shape, num_nodes ), fill_nan=False, dtype = fates_r8)


    for i in range(n_hypool_ag + n_hypool_troot):
        if i < n_hypool_ag:  # Adjusted condition for 0-based indexing
            z_node[:,:,i]       = z_node_ag[:,:,i]
            v_node[:,:,i]       = v_ag[:,:,i]
        else:
            z_node[:,:,i]       = z_node_troot
            v_node[:,:,i]       = v_troot

    i = n_hypool_ag + n_hypool_troot - 1  # Adjust i for the next operations, -1 for 0-based indexing

    # l_aroot_layer_CC has the dimension of nsites, ncohorts, nlevrhiz
    # l_aroot_layer    has the dimension of nsites, nlevrhiz -->
    # need to be expanded in ncohorts dim = 1

    mask             = (l_aroot_layer_CC > nearzero).bool()
    aroot_frac_plant = torch.where(mask,l_aroot_layer_CC / l_aroot_layer.unsqueeze(dim = 1),0.0)

    for j in range(nlevrhiz):  # Adjust for Python's 0-based indexing
        for k in range(n_hypool_aroot + nshell):  # +1 because range() is exclusive on the stop value
            i += 1
            z_node[:, :, i] = (-zi_rhiz[:, j] + 0.5 * dz_rhiz[:, j]).unsqueeze(dim=1)

            if k == 0:
                v_node[:,:,i]       = v_aroot_layer[:,:,j]
            else:
                kshell              = k - 1  # Adjust for 0-based indexing and k starting from 1
                # Calculate volume of the Rhizosphere for a single plant
                v_node[:,:,i]       = (v_shell[:, j,kshell]).unsqueeze(dim=1) * aroot_frac_plant[:,:,j]

    return z_node, v_node
#=======================================================================================================
def Gethnode( mask_plant_media,
              th_node_init,
              th_plant,
              h2osoi_liqvol_shell,
              pm_node,
              hydr_media_id):
    #=================================================================================================
    th_node_init[:, :, mask_plant_media] = th_plant
    # Be very cautious here in case nshells >1. need to test if this still works
    #============================================================================
    mask_rhiz                     = pm_node == hydr_media_id['rhiz_p_media']
    th_node_init[:, :, mask_rhiz] = (h2osoi_liqvol_shell.squeeze(dim=-1)).unsqueeze(dim=1) # of sites_hydr
    return th_node_init
#=======================================================================================================================
def GethPsiftcnode(z_node,
                   pm_node,
                   mask_plant_media,
                   wrf_plant,wkf_plant,
                   wrf_soil, wkf_soil,
                   th_node,
                   hydr_media_id):

    psi_node = torch.zeros_like(th_node)
    ftc_node = torch.zeros_like(th_node)

    # SOIL
    #==============================================================================================================
    mask_rhiz_media = pm_node ==  hydr_media_id['rhiz_p_media']
    psi_node[..., mask_rhiz_media] = wrf_soil.psi_from_th(th_node[..., mask_rhiz_media])
    ftc_node[..., mask_rhiz_media] = wkf_soil.ftc_from_psi(psi_node[..., mask_rhiz_media])

    # PLANT
    #==============================================================================================================
    psi_node[..., mask_plant_media] = wrf_plant.psi_from_th(th_node[..., mask_plant_media])
    ftc_node[..., mask_plant_media] = wkf_plant['plant_media'].ftc_from_psi(psi_node[..., mask_plant_media])
    h_node = mpa_per_pa * denh2o * grav_earth * z_node+ psi_node
    return h_node, psi_node, ftc_node
#=======================================================================================================
def GetQflux(conn_dn,
             conn_up,
             kmax_dn,
             kmax_up,
             h_node,
             ftc_node,
             do_upstream_k= True):

    idxs_dn = conn_dn - 1
    idxs_up = conn_up - 1
    h_node_dn   = h_node[..., idxs_dn]
    h_node_up   = h_node[..., idxs_up]
    ftc_node_dn = ftc_node[...,idxs_dn]
    ftc_node_up = ftc_node[...,idxs_up]
    k_eff = GetKAndDKDPsi(kmax_dn,
                          kmax_up,
                          h_node_dn,
                          h_node_up,
                          ftc_node_dn,
                          ftc_node_up,
                          do_upstream_k)

    q_flux = k_eff * (h_node_up - h_node_dn)

    return q_flux
#=======================================================================================================
def GetResidual_th(qtop, q_flux, conn_dn, conn_up,v_node, x, th_node_prev, dtime ):
    # num_nodes = v_node.shape[-1]
    # # for k in range(num_nodes):
    # #     # ! This is the storage gained from previous newton iterations.
    # #     residual[:,:,k] = residual[:,:,k] + denh2o * v_node[:,:,k] * (x[:,:,k] - th_node_prev[:,:,k])/dtime
    # residual =  denh2o * v_node * (x - th_node_prev) / dtime
    #
    # # ! Add fluxes at current time to the residual
    # num_connections = q_flux.shape[-1]
    # for icnx in range(num_connections):
    #     id_dn = conn_dn[icnx] - 1
    #     id_up = conn_up[icnx] - 1
    #     residual[:,:, id_dn] = residual[:,:,id_dn] - q_flux[:,:,icnx]
    #     residual[:,:, id_up] = residual[:,:,id_up] + q_flux[:,:,icnx]
    #
    # residual[:,:,0] = residual[:,:,0] + qtop
    # Calculate the residual without looping over nodes
    residual = denh2o * v_node * (x - th_node_prev) / dtime

    # Adjust residuals using advanced indexing in PyTorch
    residual.index_add_(-1, conn_dn - 1, -q_flux)
    residual.index_add_(-1, conn_up - 1, q_flux)

    # Add qtop to the first node
    residual[...,0] += qtop
    f = residual
    return f
#=======================================================================================================
def get_guess_th(dims,
             th_node_init,
             th_ag,
             th_troot,
             th_aroot  ,
             h2osoi_liqvol_shell):

    n_hypool_ag    = dims['n_hypool_ag']
    n_hypool_troot = dims['n_hypool_troot']
    n_hypool_aroot = dims['n_hypool_aroot']
    nlevrhiz       = dims['nlevrhiz']
    nshell         = dims['nshell']

    for i in range(n_hypool_ag + n_hypool_troot):
        if i < n_hypool_ag:  # Adjusted condition for 0-based indexing
            th_node_init[:,:,i] = th_ag[:,:,i]
        else:
            th_node_init[:,:,i] = th_troot

    i = n_hypool_ag + n_hypool_troot - 1  # Adjust i for the next operations, -1 for 0-based indexing

    for j in range(nlevrhiz):  # Adjust for Python's 0-based indexing
        for k in range(n_hypool_aroot + nshell):  # +1 because range() is exclusive on the stop value
            i += 1
            if k == 0:
                th_node_init[:,:,i] = th_aroot[:,:,j]
            else:
                kshell              = k - 1  # Adjust for 0-based indexing and k starting from 1
                # Calculate volume of the Rhizosphere for a single plant
                th_node_init[:,:,i] = (h2osoi_liqvol_shell[:,j,kshell]).unsqueeze(dim=1) # of sites_hydr

    return th_node_init
#=======================================================================================================================
def UpdatePlantPsiFTCFromTheta(th_plant,
                               wrf_plant,
                               wkf_plant,
                               ):

    # ! This subroutine updates the potential and the fractional
    # ! of total conductivity based on the relative water
    # ! content
    # ! Arguments
    # type(ed_cohort_type),intent(inout), target :: ccohort

    # ! Update Psi and FTC in above-ground compartments
    # ! -----------------------------------------------------------------------------------
    #================================================================================
    psi_plant  = wrf_plant.psi_from_th(th_plant)
    ftc_plant  = wkf_plant['plant_media'].ftc_from_psi(psi_plant)
    # ===============================================================================

    return psi_plant, ftc_plant
#=======================================================================================================================
# def Update_H2oSoi(  dz_sisl,
#                     eff_porosity_sl,
#                     h2o_liq_sisl,
#                     dz_rhiz,
#                     map_r2s,
#                     h2osoi_liqvol_shell,
#                     mean_type):
#     _, nlevrhiz, nshell   = h2osoi_liqvol_shell.shape
#
#     eff_por = AggBCToRhiz(map_r2s, eff_porosity_sl,dz_sisl, mean_type)
#
#     # ! [kg/m2] / ([m] * [kg/m3]) = [m3/m3]
#     h2osoi_liqvol = torch.minimum(eff_por, torch.sum(h2o_liq_sisl.unsqueeze(dim=-1) * map_r2s.unsqueeze(0), dim = -1)/(dz_rhiz*denh2o))
#     h2osoi_liqvol_shell[:,:,:nshell] = h2osoi_liqvol.unsqueeze(dim = -1)
#     return h2osoi_liqvol_shell

#=======================================================================================================================
#==============================================================================================================
#==============================================================================================================
#==============================================================================================================
# PHOTOSYNTHESIS MODEL
########################################################################################################
def quadratic_min(aquad,bquad,cquad):
    #Description: Solve for the minimum root of a quadratic equation
    #Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          #! Computing (Cambridge University Press, Cambridge), pp. 145.


    #aquad, bquad, cquad are the terms of a quadratic equation
    #r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1   = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)
    #RUN CHECK#
    #r2[torch.where(torch.isnan(r2))[0]] = 1.e36 * ( 1 - mask[torch.where(torch.isnan(r2))[0]])
    # r1 = (-0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)) )/ aquad
    # r2 = (-0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)))/ aquad


    return torch.min(r1,r2)
########################################################################################################
def quadratic_max(aquad,bquad,cquad):
    # Description: Solve for the maximum root of a quadratic equation
    # Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          # ! Computing (Cambridge University Press, Cambridge), pp. 145.


    # Inputs : aquad, bquad, cquad are the terms of a quadratic equation
    # outputs: r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1 = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)
    #RUN CHECK#
    #r2[torch.where(torch.isnan(r2))[0]] = 1.e36 * ( 1 - mask[torch.where(torch.isnan(r2))[0]])
    # r1 = (-0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)) )/ aquad
    # r2 = (-0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)))/ aquad

    return torch.max(r1,r2)
########################################################################################################
def QuadraticRootsSridharachary(a, b, c):
    """
    Compute roots for the quadratic equation Ax^2 + Bx + C = 0 using tensors.

    Args:
    a (torch.Tensor): Coefficient of x^2
    b (torch.Tensor): Coefficient of x
    c (torch.Tensor): Constant term

    Returns:
    torch.Tensor: Roots of the quadratic equation
    """
    nearzero = 1e-10  # Define a small threshold to handle numerical stability
    root1   = torch.zeros_like(a)
    root2   = torch.zeros_like(a)

    # Handle linear case where a is near zero
    linear_mask     = torch.abs(a) < nearzero
    non_linear_mask = ~linear_mask
    non_zero_b      = torch.abs(b) > nearzero

    root2[linear_mask & non_zero_b] = -c[linear_mask & non_zero_b] / b[linear_mask & non_zero_b]

    # Calculate discriminant
    d = b ** 2 - 4 * a * c

    # Compute roots based on the discriminant
    das                     = torch.sqrt(torch.abs(d))
    real_roots_mask         = d > nearzero
    double_root_mask        = torch.abs(d) <= nearzero
    imaginary_roots_mask    = d < -nearzero

    # Real roots
    root1[non_linear_mask & real_roots_mask] = (-b[non_linear_mask & real_roots_mask] +
                                                das[non_linear_mask & real_roots_mask]) / (2 * a[non_linear_mask & real_roots_mask])
    root2[non_linear_mask & real_roots_mask] = (-b[non_linear_mask & real_roots_mask] -
                                                das[non_linear_mask & real_roots_mask]) / (2 * a[non_linear_mask & real_roots_mask])

    # Repeated real roots
    root1[non_linear_mask & double_root_mask] = -b[non_linear_mask & double_root_mask] / (2 * a[non_linear_mask & double_root_mask])
    root2[non_linear_mask & double_root_mask] = root1[non_linear_mask & double_root_mask]

    # Handling imaginary roots scenario by raising an error
    if torch.any(imaginary_roots_mask):
        raise ValueError("Error, imaginary roots detected in quadratic solve")

    return root1, root2
# a = torch.tensor([1.0, 2.0, 1.0])
# b = torch.tensor([5.0, 8.0, 3.0])
# c = torch.tensor([6.0, 0.0, 2.0])
#
# root1, root2 = QuadraticRootsSridharachary(a, b, c)
# print("Root1:", root1)
# print("Root2:", root2)
########################################################################################################
def ft1_f(tl,ha):
    # DESCRIPTION:photosynthesis temperature response
    # Copied from: FATES

    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # ha: activation energy in photosynthesis temperature function (J/mol)

    # outputs: parameter scaled to leaf temperature (tl)

    return torch.exp(ha/(rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm + 25)) *(1.0-(t_water_freeze_k_1atm + 25.0)/tl))
########################################################################################################

def fth25_f(hd,se):
    # Description:scaling factor for photosynthesis temperature inhibition
    # Copied from: FATES

    # Inputs :
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)

    # outputs: parameter scaled to leaf temperature (tl)
    # return 1.0 + torch.exp(torch.tensor(-hd + se * (t_water_freeze_k_1atm + 25.0)) /
    #                    (rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm+25.0)))
    return 1.0 + torch.exp((-hd + se * (t_water_freeze_k_1atm + 25.0)) /
                          (rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm+25.0)))


########################################################################################################

def fth_f(tl,hd,se,scaleFactor):
    # Description:photosynthesis temperature inhibition
    # Copied from: FATES
    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)
    #scaleFactor  ! scaling factor for high temp inhibition (25 C = 1.0)

    return scaleFactor / (1.0 + torch.exp((-hd+se*tl) / (rgas_J_K_kmol * 1.0e-3 * tl)))

########################################################################################################
def QSat(tempk):
    # Description:Computes saturation mixing ratio and the change in saturation
    # Copied from: CLM5.0

    # Parameters for derivative:water vapor
    a0 =  6.11213476;      a1 =  0.444007856     ; a2 =  0.143064234e-01 ; a3 =  0.264461437e-03
    a4 =  0.305903558e-05; a5 =  0.196237241e-07;  a6 =  0.892344772e-10 ; a7 = -0.373208410e-12
    a8 =  0.209339997e-15

    # Parameters For ice (temperature range -75C-0C)
    c0 =  6.11123516;      c1 =  0.503109514;     c2 =  0.188369801e-01; c3 =  0.420547422e-03
    c4 =  0.614396778e-05; c5 =  0.602780717e-07; c6 =  0.387940929e-09; c7 =  0.149436277e-11
    c8 =  0.262655803e-14;

    # Inputs:
    # tempk: temperature in kelvin
    # RH   : Relative humidty in fraction

    #outputs:
    # veg_esat: saturated vapor pressure at tempk (pa)
    # air_vpress: air vapor pressure (pa)
    td = torch.clamp(tempk - t_water_freeze_k_1atm, min = -75.0, max = 100.0)


    mask = td >= 0.0
    mask = mask.type(torch.uint8)

    veg_esat = (a0 + td*(a1 + td*(a2 + td*(a3 + td*(a4 + td*(a5 + td*(a6 + td*(a7 + td*a8)))))))) * mask + \
               (c0 + td*(c1 + td*(c2 + td*(c3 + td*(c4 + td*(c5 + td*(c6 + td*(c7 + td*c8)))))))) * (1 - mask)

    veg_esat = veg_esat * 100.0           # pa
    # air_vpress = RH * veg_esat            #RH as fraction
    return veg_esat
#=======================================================================================================================
def GetCanopytopRates(vcmax25top_ft): #c3c4_path_index,
    jmax25top_ft              = 1.67 * vcmax25top_ft
    co2_rcurve_islope25top_ft = 20000 * vcmax25top_ft
    lmr_25top_ft = lmr25top_ft_extract(vcmax25top_ft) #c3c4_path_index,
    return jmax25top_ft, co2_rcurve_islope25top_ft, lmr_25top_ft
#=======================================================================================================================
def GetCanopyGasParameters(
                           veg_tempk,
                           air_vpress,
                           veg_esat,
                           kc25, ko25, cp25,
                           ):

    # Description: calculates the specific Michaelis Menten Parameters (pa) for CO2 and O2, as well as
    # the CO2 compentation point.
    # Copied from: FATES

    # Inputs:
    # can_press          : Air pressure within the canopy (Pa)
    # can_o2_partialpress: Partial press of o2 in the canopy (Pa
    # veg_tempk          : The temperature of the vegetation (K)
    # air_tempk          : Temperature of canopy air (K)
    # air_vpress         : Vapor pressure of canopy air (Pa)
    # veg_esat           : Saturated vapor pressure at veg surf (Pa)
    # rb                 : Leaf Boundary layer resistance (s/m)

    # Outputs:
    # mm_kco2   :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2    :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint:CO2 compensation point (Pa)
    # cf        :conversion factor between molar form and velocity form of conductance and resistance: [umol/m3]
    # gb_mol    :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair     :vapor pressure of air, constrained (Pa)



    # mask = (veg_tempk > 150.0) & (veg_tempk < 350.0)
    # mask = mask.bool()
    # # print((1-mask.type(torch.uint8)).sum())
    #
    # mm_kco2    = torch.where(mask, kc25 * ft1_f(veg_tempk, kcha) , 1.0)
    # mm_ko2     = torch.where(mask, ko25 * ft1_f(veg_tempk, koha) , 1.0)
    # co2_cpoint = torch.where(mask, cp25 * ft1_f(veg_tempk, cpha) , 1.0)
    mask = (veg_tempk <= 150.0) | (veg_tempk >= 350.0)
    if torch.any(mask):
        raise ValueError("veg_tempk outside range [150 - 350]")

    mm_kco2    = kc25 * ft1_f(veg_tempk, kcha)
    mm_ko2     = ko25 * ft1_f(veg_tempk, koha)
    co2_cpoint = cp25 * ft1_f(veg_tempk, cpha)
    ceair      = torch.clamp(air_vpress, min=0.05 * veg_esat, max=veg_esat)

    return mm_kco2, mm_ko2, co2_cpoint, ceair
########################################################################################################
def lmr25top_ft_extract(vcmax25top_ft):# c3c4_path_index,
    # Description: calculates the canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # for C3 plants : lmr_25top_ft = 0.015 vcmax25top_ft
    # for C4 plants : lmr_25top_ft = 0.025 vcmax25top_ft

    # Inputs:
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)
    # vcmax25top_ft  : canopy top maximum rate of carboxylation at 25C for this pft (umol CO2/m**2/s)

    # Outputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)

    # mask = c3c4_path_index == c3_path_index
    # mask = mask.type(torch.uint8)
    lmr_25top_ft = 0.015 * vcmax25top_ft # (0.015 * vcmax25top_ft) * mask + (0.025 * vcmax25top_ft) * (1 - mask)
    return lmr_25top_ft

########################################################################################################
def LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler,veg_tempk):
    # Description :  Base maintenance respiration rate for plant tissues maintresp_leaf_ryan1991_baserate
    # M. Ryan, 1991. Effects of climate change on plant respiration. It rescales the canopy top leaf maint resp
    # rate at 25C to the vegetation temperature (veg_tempk)

    # Inputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # nscaler        : leaf nitrogen scaling coefficient (assumed here as 1)
    # veg_tempk      : vegetation temperature
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)

    # Outputs:
    # lmr    : Leaf Maintenance Respiration  (umol CO2/m**2/s)

    lmr25 = lmr25top_ft * nscaler
    lmr   = lmr25 * ft1_f(veg_tempk, lmrha) * fth_f(veg_tempk, lmrhd, lmrse, lmrc)

    return lmr
#=======================================================================================================================
def LeafLayerMaintenanceRespiration_Ryan_1991(maintresp_leaf_ryan1991_baserate,
                                              c3c4_path_index,
                                              lnc_top,
                                              nscaler,
                                              veg_tempk):

    # ! -----------------------------------------------------------------------
    # ! Base maintenance respiration rate for plant tissues maintresp_leaf_ryan1991_baserate
    # ! M. Ryan, 1991. Effects of climate change on plant respiration.
    # ! Ecological Applications, 1(2), 157-167.
    # ! Original expression is br = 0.0106 molC/(molN h)
    # ! Conversion by molecular weights of C and N gives 2.525e-6 gC/(gN s)
    # ! Which is the default value of maintresp_nonleaf_baserate
    #
    # ! Arguments
    # real(r8), intent(in)  :: lnc_top      ! Leaf nitrogen content per unit area at canopy top [gN/m2]
    # real(r8), intent(in)  :: nscaler      ! Scale for leaf nitrogen profile
    # integer,  intent(in)  :: ft           ! (plant) Functional Type Index
    # real(r8), intent(in)  :: veg_tempk    ! vegetation temperature
    # real(r8), intent(out) :: lmr          ! Leaf Maintenance Respiration  (umol CO2/m**2/s)
    #
    # ! Locals
    # real(r8) :: lmr25   ! leaf layer: leaf maintenance respiration rate at 25C (umol CO2/m**2/s)
    # real(r8) :: lmr25top  ! canopy top leaf maint resp rate at 25C for this pft (umol CO2/m**2/s)
    # integer :: c3c4_path_index    ! Index for which photosynthetic pathway
    #
    # ! Parameter
    # real(r8), parameter :: lmrha = 46390._r8    ! activation energy for lmr (J/mol)
    # real(r8), parameter :: lmrhd = 150650._r8   ! deactivation energy for lmr (J/mol)
    # real(r8), parameter :: lmrse = 490._r8      ! entropy term for lmr (J/mol/K)
    # real(r8), parameter :: lmrc = 1.15912391_r8 ! scaling factor for high
    # ! temperature inhibition (25 C = 1.0)

    lmr25top = maintresp_leaf_ryan1991_baserate * (1.5 ** ((25. - 20.)/10.))
    lmr25top = lmr25top * lnc_top / (umolC_to_kgC * g_per_kg)


    # ! Part I: Leaf Maintenance respiration: umol CO2 / m**2 [leaf] / s
    # ! ----------------------------------------------------------------------------------
    lmr25 = lmr25top * nscaler

    # ! photosynthetic pathway: 0. = c4, 1. = c3
    # ! temperature sensitivity of C4 plants
    lmr = lmr25 * 2.**((veg_tempk-(tfrz+25.))/10.)
    lmr = lmr / (1. + torch.exp( 1.3*(veg_tempk-(tfrz+55.)) ))

    mask = (c3c4_path_index == c3_path_index).bool()
    # ! temperature sensitivity of C3 plants
    lmr = torch.where(mask,
                      lmr25 * ft1_f(veg_tempk, lmrha) * fth_f(veg_tempk, lmrhd, lmrse, lmrc),
                      lmr)


    # ! Any hydrodynamic limitations could go here, currently none
    # ! lmr = lmr * (nothing)
    return lmr
#=======================================================================================================================
def LeafLayerMaintenanceRespiration_Atkin_etal_2017(maintresp_leaf_atkin2017_baserate,
                                                    maintresp_leaf_vert_scaler_coeff1, 
                                                    maintresp_leaf_vert_scaler_coeff2,
                                                    lnc_top, 
                                                    cumulative_lai, 
                                                    vcmax25top,     
                                                    veg_tempk,
                                                    tgrowth,        
                                                    ):

    # use FatesConstantsMod, only : tfrz => t_water_freeze_k_1atm
    # use FatesConstantsMod, only : umolC_to_kgC
    # use FatesConstantsMod, only : g_per_kg
    # use FatesConstantsMod, only : lmr_b
    # use FatesConstantsMod, only : lmr_c
    # use FatesConstantsMod, only : lmr_TrefC
    # use FatesConstantsMod, only : lmr_r_1
    # use FatesConstantsMod, only : lmr_r_2
    # use EDPftvarcon      , only : EDPftvarcon_inst

    # ! Arguments
    # real(r8), intent(in)  :: lnc_top          ! Leaf nitrogen content per unit area at canopy top [gN/m2]
    # integer,  intent(in)  :: ft               ! (plant) Functional Type Index
    # real(r8), intent(in)  :: vcmax25top       ! top of canopy vcmax
    # real(r8), intent(in)  :: cumulative_lai   ! cumulative lai above the current leaf layer
    # real(r8), intent(in)  :: veg_tempk        ! vegetation temperature  (degrees K)
    # real(r8), intent(in)  :: tgrowth          ! lagged vegetation temperature averaged over acclimation timescale (degrees K)
    # real(r8), intent(out) :: lmr              ! Leaf Maintenance Respiration  (umol CO2/m**2/s)

    # ! Locals
    # real(r8) :: lmr25   ! leaf layer: leaf maintenance respiration rate at 25C (umol CO2/m**2/s)
    # real(r8) :: r_0     ! base respiration rate, PFT-dependent (umol CO2/m**2/s)
    # real(r8) :: r_t_ref ! acclimated ref respiration rate (umol CO2/m**2/s)
    # real(r8) :: lmr25top  ! canopy top leaf maint resp rate at 25C for this pft (umol CO2/m**2/s)
    # 
    # real(r8) :: rdark_scaler ! negative exponential scaling of rdark
    # real(r8) :: kn           ! decay coefficient

    # ! parameter values of r_0 as listed in Atkin et al 2017: (umol CO2/m**2/s) 
    # ! Broad-leaved trees  1.7560
    # ! Needle-leaf trees   1.4995
    # ! Shrubs              2.0749
    # ! C3 herbs/grasses    2.1956
    # ! In the absence of better information, we use the same value for C4 grasses as C3 grasses.

    # ! r_0 currently put into the EDPftvarcon_inst%dev_arbitrary_pft
    # ! all figs in Atkin et al 2017 stop at zero Celsius so we will assume acclimation is fixed below that
    r_0 = maintresp_leaf_atkin2017_baserate

    # ! This code uses the relationship between leaf N and respiration from Atkin et al 
    # ! for the top of the canopy, but then scales through the canopy based on a rdark_scaler.
    # ! To assume proportionality with N through the canopy following Lloyd et al. 2010, use the
    # ! default parameter value of 2.43, which results in the scaling of photosynthesis and respiration
    # ! being proportional through the canopy. To have a steeper decrease in respiration than photosynthesis
    # ! this number can be smaller. There is some observational evidence for this being the case
    # ! in Lamour et al. 2023. 

    kn = decay_coeff_vcmax(vcmax25top, 
                           maintresp_leaf_vert_scaler_coeff1, 
                           maintresp_leaf_vert_scaler_coeff2)

    rdark_scaler = torch.exp(-kn * cumulative_lai)
    t_clamp = torch.clamp(tgrowth - tfrz , min = 0.0)
    r_t_ref = torch.clamp(rdark_scaler * (r_0 + lmr_r_1 * lnc_top + lmr_r_2 * t_clamp), min = 0.0)
    # max(0._r8, rdark_scaler * (r_0 + lmr_r_1 * lnc_top + lmr_r_2 * max(0._r8, (tgrowth - tfrz) )) )

    mask = (r_t_ref == 0.0).bool()
    if torch.any(mask):
        print("Warning message: Rdark is negative for some points")
    # Rdark is negative at this temperature and is capped at 0. tgrowth (C)
    lmr = r_t_ref * torch.exp(lmr_b * (veg_tempk - tfrz - lmr_TrefC) + lmr_c *
                              ((veg_tempk-tfrz)**2 - lmr_TrefC**2))

    return lmr
#=======================================================================================================================
def  LeafLayerBiophysicalRates(parsun_per_la,
                               vcmax25top_ft,
                               jmax25top_ft,
                               co2_rcurve_islope25top_ft,
                               vcmaxha,vcmaxhd,vcmaxse,
                               jmaxha,jmaxhd, jmaxse,
                               nscaler,
                               veg_tempk,
                               btran,
                               ):

    # ! ---------------------------------------------------------------------------------
    # ! This subroutine calculates the localized rates of several key photosynthesis
    # ! rates.  By localized, we mean specific to the plant type and leaf layer,
    # ! which factors in leaf physiology, as well as environmental effects.
    # ! This procedure should be called prior to iterative solvers, and should
    # ! have pre-calculated the reference rates for the pfts before this.
    # !
    # ! The output biophysical rates are:
    # ! vcmax: maximum rate of carboxilation,
    # ! jmax: maximum electron transport rate,
    # ! co2_rcurve_islope: initial slope of CO2 response curve (C4 plants)
    # ! ---------------------------------------------------------------------------------

    # use EDPftvarcon         , only : EDPftvarcon_inst
    #
    # ! Arguments
    # ! ------------------------------------------------------------------------------
    #
    # real(r8), intent(in) :: parsun_per_la   ! PAR absorbed per sunlit leaves for this layer
    # integer,  intent(in) :: ft              ! (plant) Functional Type Index
    # real(r8), intent(in) :: nscaler         ! Scale for leaf nitrogen profile
    # real(r8), intent(in) :: vcmax25top_ft   ! canopy top maximum rate of carboxylation at 25C
    # ! for this pft (umol CO2/m**2/s)
    # real(r8), intent(in) :: jmax25top_ft    ! canopy top maximum electron transport rate at 25C
    # ! for this pft (umol electrons/m**2/s)
    # real(r8), intent(in) :: co2_rcurve_islope25top_ft ! initial slope of CO2 response curve
    # ! (C4 plants) at 25C, canopy top, this pft
    # real(r8), intent(in) :: veg_tempk           ! vegetation temperature
    # real(r8), intent(in) :: dayl_factor         ! daylength scaling factor (0-1)
    # real(r8), intent(in) :: t_growth            ! T_growth (short-term running mean temperature) (K)
    # real(r8), intent(in) :: t_home              ! T_home (long-term running mean temperature) (K)
    # real(r8), intent(in) :: btran           ! transpiration wetness factor (0 to 1)
    #
    # real(r8), intent(out) :: vcmax             ! maximum rate of carboxylation (umol co2/m**2/s)
    # real(r8), intent(out) :: jmax              ! maximum electron transport rate
    # ! (umol electrons/m**2/s)
    # real(r8), intent(out) :: co2_rcurve_islope ! initial slope of CO2 response curve (C4 plants)
    #
    # ! Locals
    # ! -------------------------------------------------------------------------------
    # real(r8) :: vcmax25             ! leaf layer: maximum rate of carboxylation at 25C
    # ! (umol CO2/m**2/s)
    # real(r8) :: jmax25              ! leaf layer: maximum electron transport rate at 25C
    # ! (umol electrons/m**2/s)
    # real(r8) :: co2_rcurve_islope25 ! leaf layer: Initial slope of CO2 response curve
    # ! (C4 plants) at 25C
    # integer :: c3c4_path_index      ! Index for which photosynthetic pathway
    # real(r8) :: dayl_factor_local   ! Local version of daylength factor
    #
    # ! Parameters
    # ! ---------------------------------------------------------------------------------
    # real(r8) :: vcmaxha        ! activation energy for vcmax (J/mol)
    # real(r8) :: jmaxha         ! activation energy for jmax (J/mol)
    # real(r8) :: vcmaxhd        ! deactivation energy for vcmax (J/mol)
    # real(r8) :: jmaxhd         ! deactivation energy for jmax (J/mol)
    # real(r8) :: vcmaxse        ! entropy term for vcmax (J/mol/K)
    # real(r8) :: jmaxse         ! entropy term for jmax (J/mol/K)
    # real(r8) :: t_growth_celsius ! average growing temperature
    # real(r8) :: t_home_celsius   ! average home temperature
    # real(r8) :: jvr            ! ratio of Jmax25 / Vcmax25
    # real(r8) :: vcmaxc         ! scaling factor for high temperature inhibition (25 C = 1.0)
    # real(r8) :: jmaxc          ! scaling factor for high temperature inhibition (25 C = 1.0)
    # ! update the daylength factor local variable if the switch is on

    if torch.all(parsun_per_la <=0):
        vcmax = 0.0
        jmax  = 0.0
        co2_rcurve_islope = 0.0
    else:
        vcmax25 = vcmax25top_ft * nscaler
        jmax25  = jmax25top_ft * nscaler
        co2_rcurve_islope25 = co2_rcurve_islope25top_ft * nscaler

        vcmaxc = fth25_f(vcmaxhd, vcmaxse)
        jmaxc  = fth25_f(jmaxhd, jmaxse)

        vcmax             = vcmax25 * ft1_f(veg_tempk, vcmaxha) * fth_f(veg_tempk, vcmaxhd, vcmaxse, vcmaxc)
        jmax              = jmax25 * ft1_f(veg_tempk, jmaxha) * fth_f(veg_tempk, jmaxhd, jmaxse, jmaxc)
        co2_rcurve_islope = co2_rcurve_islope25 * 2.**((veg_tempk-(tfrz+25.))/10.)

        mask  = (parsun_per_la <= 0.)
        mask  = mask.type(torch.uint8)

        vcmax = mask * 0. + (1-mask)* vcmax
        jmax  = mask * 0. + (1-mask)* jmax
        co2_rcurve_islope = mask * 0.+ (1-mask) * co2_rcurve_islope

    # ! Adjust for water limitations
    vcmax = vcmax * btran

    return vcmax, jmax, co2_rcurve_islope
#=======================================================================================================================
def  LeafLayerBiophysicalRatesUpdated( photo_tempsens_model,
                                       dayl_switch,
                                       parsun_per_la,
                                       vcmax25top_ft,
                                       jmax25top_ft,
                                       co2_rcurve_islope25top_ft,
                                       vcmaxha,vcmaxhd,vcmaxse,
                                       jmaxha,jmaxhd, jmaxse,
                                       nscaler,
                                       veg_tempk,
                                       dayl_factor,
                                       t_growth,
                                       t_home,
                                       btran,
                                       c3c4_path_index):

    # ! ---------------------------------------------------------------------------------
    # ! This subroutine calculates the localized rates of several key photosynthesis
    # ! rates.  By localized, we mean specific to the plant type and leaf layer,
    # ! which factors in leaf physiology, as well as environmental effects.
    # ! This procedure should be called prior to iterative solvers, and should
    # ! have pre-calculated the reference rates for the pfts before this.
    # !
    # ! The output biophysical rates are:
    # ! vcmax: maximum rate of carboxilation,
    # ! jmax: maximum electron transport rate,
    # ! co2_rcurve_islope: initial slope of CO2 response curve (C4 plants)
    # ! ---------------------------------------------------------------------------------

    # use EDPftvarcon         , only : EDPftvarcon_inst
    #
    # ! Arguments
    # ! ------------------------------------------------------------------------------
    #
    # real(r8), intent(in) :: parsun_per_la   ! PAR absorbed per sunlit leaves for this layer
    # integer,  intent(in) :: ft              ! (plant) Functional Type Index
    # real(r8), intent(in) :: nscaler         ! Scale for leaf nitrogen profile
    # real(r8), intent(in) :: vcmax25top_ft   ! canopy top maximum rate of carboxylation at 25C
    # ! for this pft (umol CO2/m**2/s)
    # real(r8), intent(in) :: jmax25top_ft    ! canopy top maximum electron transport rate at 25C
    # ! for this pft (umol electrons/m**2/s)
    # real(r8), intent(in) :: co2_rcurve_islope25top_ft ! initial slope of CO2 response curve
    # ! (C4 plants) at 25C, canopy top, this pft
    # real(r8), intent(in) :: veg_tempk           ! vegetation temperature
    # real(r8), intent(in) :: dayl_factor         ! daylength scaling factor (0-1)
    # real(r8), intent(in) :: t_growth            ! T_growth (short-term running mean temperature) (K)
    # real(r8), intent(in) :: t_home              ! T_home (long-term running mean temperature) (K)
    # real(r8), intent(in) :: btran           ! transpiration wetness factor (0 to 1)
    #
    # real(r8), intent(out) :: vcmax             ! maximum rate of carboxylation (umol co2/m**2/s)
    # real(r8), intent(out) :: jmax              ! maximum electron transport rate
    # ! (umol electrons/m**2/s)
    # real(r8), intent(out) :: co2_rcurve_islope ! initial slope of CO2 response curve (C4 plants)
    #
    # ! Locals
    # ! -------------------------------------------------------------------------------
    # real(r8) :: vcmax25             ! leaf layer: maximum rate of carboxylation at 25C
    # ! (umol CO2/m**2/s)
    # real(r8) :: jmax25              ! leaf layer: maximum electron transport rate at 25C
    # ! (umol electrons/m**2/s)
    # real(r8) :: co2_rcurve_islope25 ! leaf layer: Initial slope of CO2 response curve
    # ! (C4 plants) at 25C
    # integer :: c3c4_path_index      ! Index for which photosynthetic pathway
    # real(r8) :: dayl_factor_local   ! Local version of daylength factor
    #
    # ! Parameters
    # ! ---------------------------------------------------------------------------------
    # real(r8) :: vcmaxha        ! activation energy for vcmax (J/mol)
    # real(r8) :: jmaxha         ! activation energy for jmax (J/mol)
    # real(r8) :: vcmaxhd        ! deactivation energy for vcmax (J/mol)
    # real(r8) :: jmaxhd         ! deactivation energy for jmax (J/mol)
    # real(r8) :: vcmaxse        ! entropy term for vcmax (J/mol/K)
    # real(r8) :: jmaxse         ! entropy term for jmax (J/mol/K)
    # real(r8) :: t_growth_celsius ! average growing temperature
    # real(r8) :: t_home_celsius   ! average home temperature
    # real(r8) :: jvr            ! ratio of Jmax25 / Vcmax25
    # real(r8) :: vcmaxc         ! scaling factor for high temperature inhibition (25 C = 1.0)
    # real(r8) :: jmaxc          ! scaling factor for high temperature inhibition (25 C = 1.0)
    # ! update the daylength factor local variable if the switch is on
    if (dayl_switch == itrue):
        dayl_factor_local = dayl_factor
    else:
        dayl_factor_local = 1.0

    # ! Vcmax25top was already calculated to derive the nscaler function
    vcmax25 = vcmax25top_ft * nscaler * dayl_factor_local

    if photo_tempsens_model == "photosynth_acclim_model_none":
        jmax25  = jmax25top_ft * nscaler * dayl_factor_local
    elif photo_tempsens_model == "photosynth_acclim_model_kumarathunge_etal_2019":
    # !Kumarathunge et al. temperature acclimation, Thome=30-year running mean
        t_growth_celsius    = t_growth-t_water_freeze_k_1atm
        t_home_celsius      = t_home-t_water_freeze_k_1atm
        vcmaxha             = (42.6 + (1.14*t_growth_celsius))*1e3
        jmaxha              = 40.71*1e3
        vcmaxhd             = 200.*1e3
        jmaxhd              = 200.*1e3
        vcmaxse             = (645.13 - (0.38*t_growth_celsius))
        jmaxse              = 658.77 - (0.84*t_home_celsius) - 0.52*(t_growth_celsius-t_home_celsius)
        jvr                 = 2.56 - (0.0375*t_home_celsius)-(0.0202*(t_growth_celsius-t_home_celsius))
        jmax25              = vcmax25 * jvr
    else:
        raise ValueError('error, incorrect leaf photosynthesis temperature acclimation model specified')

    vcmaxc = fth25_f(vcmaxhd, vcmaxse)
    jmaxc  = fth25_f(jmaxhd, jmaxse)

    co2_rcurve_islope25 = co2_rcurve_islope25top_ft * nscaler

    mask = (c3c4_path_index == c3_path_index).bool()
    vcmax = vcmax25 * 2. ** ((veg_tempk - (tfrz + 25.)) / 10.)
    vcmax = vcmax / (1. + torch.exp(0.2 * ((tfrz + 15.) - veg_tempk)))
    vcmax = vcmax / (1. + torch.exp(0.3 * (veg_tempk - (tfrz + 40.))))
    vcmax= torch.where(mask, vcmax25 * ft1_f(veg_tempk, vcmaxha) * fth_f(veg_tempk, vcmaxhd, vcmaxse, vcmaxc),
                       vcmax)

    jmax  = jmax25 * ft1_f(veg_tempk, jmaxha) * fth_f(veg_tempk, jmaxhd, jmaxse, jmaxc)
    co2_rcurve_islope = co2_rcurve_islope25 * 2.**((veg_tempk-(tfrz+25.))/10.)

    mask  = (parsun_per_la <= 0.).bool()
    vcmax = torch.where(mask,0., vcmax)
    jmax  = torch.where(mask,0., jmax)
    co2_rcurve_islope = torch.where(mask, 0., co2_rcurve_islope)
    # ! Adjust for water limitations
    vcmax = vcmax * btran

    return vcmax, jmax, co2_rcurve_islope

########################################################################################################
def RootLayerNFixation(nfix_mresp_scfrac,
                       t_soil,
                       dtime,
                       fnrt_mr_layer,
                       ):


    # ! -------------------------------------------------------------------------------
    # ! Symbiotic N Fixation is handled via Houlton et al 2008 and Fisher et al. 2010
    # !
    # ! A unifying framework for dinitrogen fixation in the terrestrial biosphere
    # ! Benjamin Z. Houlton, Ying-Ping Wang, Peter M. Vitousek & Christopher B. Field 
    # ! Nature volume 454, pages327–330 (2008)  https://doi.org/10.1038/nature07028
    # !
    # ! Carbon cost of plant nitrogen acquisition: A mechanistic, globally applicable model
    # ! of plant nitrogen uptake, retranslocation, and fixation.  J. B. Fisher,S. Sitch,Y.
    # ! Malhi,R. A. Fisher,C. Huntingford,S.-Y. Tan. Global Biogeochemical Cycles. March
    # ! 2010 https://doi.org/10.1029/2009GB003621
    # !
    # ! ------------------------------------------------------------------------------


    # real(r8),intent(in) :: t_soil              ! Temperature of the current soil layer [degC]
    # integer,intent(in)  :: ft                  ! Functional type index
    # real(r8),intent(in) :: dtime               ! Time step length [s]
    # real(r8),intent(in) :: fnrt_mr_layer       ! Amount of maintenance respiration in the fine-roots
    # ! for all non-fixation related processes [kgC/s]

    # real(r8),intent(out) :: fnrt_mr_nfix_layer ! The added maintenance respiration due to nfixation
    # ! to be added as a surcharge to non-fixation MR [kgC]
    # real(r8),intent(out) :: nfix_layer         ! The amount of N fixed in this layer through
    # ! symbiotic activity [kgN]
    # 
    # real(r8) :: c_cost_nfix                    ! carbon cost of N fixation [kgC/kgN]
    # real(r8) :: c_spent_nfix                   ! carbon spent on N fixation, per layer [kgC/plant/timestep]
    # 
    # ! N fixation parameters from Houlton et al (2008) and Fisher et al (2010)

    # ! Amount of C spent (as part of MR respiration) on symbiotic fixation [kgC/s]
    fnrt_mr_nfix_layer  = fnrt_mr_layer * nfix_mresp_scfrac

    # ! This is the unit carbon cost for nitrogen fixation. It is temperature dependant [kgC/kgN]
    c_cost_nfix = s_fix * (torch.exp(a_fix + b_fix * (t_soil-tfrz)
                               * (1. - 0.5 * (t_soil-tfrz) / c_fix)) - 2.)

    # ! Time integrated amount of carbon spent on fixation (in this layer) [kgC/plant/layer/tstep]
    c_spent_nfix = fnrt_mr_nfix_layer  * dtime

    # ! Amount of nitrogen fixed in this layer [kgC/plant/layer/tstep]/[kgC/kgN] = [kgN/plant/layer/tstep]
    nfix_layer = c_spent_nfix / c_cost_nfix

    return fnrt_mr_nfix_layer, nfix_layer
#=======================================================================================================================
def get_guess_ci(co2_ppress):#c3c4_path_index,
    # Description: A function to get the initial guess for the intercellular leaf CO2 pressure (Ci) which is mainly
    # dependent on the classification of C3 and C4 plants.

    # Inputs:
    # c3c4_path_index: Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
    # can_co2_ppress : Partial pressure of CO2 NEAR the leaf surface (Pa)
    # mask = (c3c4_path_index == c3_path_index).bool()
    ci = init_a2l_co2_c3 * co2_ppress # torch.where(mask, init_a2l_co2_c3 * co2_ppress, init_a2l_co2_c4 * co2_ppress)
    return ci
#=======================================================================================================================
def get_guess_tleaf(air_temp):
    return air_temp + 1.0
#=======================================================================================================================
def getleaflayerAn(ci,
                je,
                co2_cpoint,
                mm_kco2,
                mm_ko2,
                vcmax,
                lmr,
                can_o2_ppress,
                parlsl_abs, # wm-2
                lai_lsl):

    if torch.any(ci - co2_cpoint) < 0:
        raise ValueError("ci - co2_cpoint need to be clamped")

    if torch.all(parlsl_abs <=0):
        anet = -lmr
    else:
        ac = vcmax * (ci - co2_cpoint) / (ci + mm_kco2 * (1.0 + can_o2_ppress / mm_ko2))
        aj = je * (ci - co2_cpoint) / (4.0 * ci + 8.0 * co2_cpoint)

        agross = quadratic_min(theta_cj_c3, -(ac + aj), ac * aj)
        anet   = agross - lmr
        # Third:  correct anet for LAI > 0.0 and parlsl_abs <=0.0
        ### added them to give the correct values
        mask = (parlsl_abs <= 0.0)
        mask = mask.type(torch.uint8)
        anet = mask * -lmr + (1-mask) * anet

    mask = (lai_lsl > 0.0)
    mask = mask.type(torch.uint8)
    anet = mask*anet + (1-mask)* 0.0

    return anet

def getCANC02press(at_leaf,
                   co2_ppress,
                   gb_mol,
                   a_gs,
                   can_press):
    if at_leaf:
        leaf_co2_ppress = co2_ppress.clone()
        can_co2_ppress = leaf_co2_ppress + h2o_co2_bl_diffuse_ratio /  gb_mol * a_gs * can_press
    else:
        can_co2_ppress  = co2_ppress
        leaf_co2_ppress = can_co2_ppress - h2o_co2_bl_diffuse_ratio / gb_mol * a_gs * can_press

    if (torch.any(can_co2_ppress) < 1e-6) or (torch.any(leaf_co2_ppress) < 1e-6):
        raise ValueError("co2 pressure below the minimum threshold")
    return can_co2_ppress, leaf_co2_ppress
#=======================================================================================================================
def getleaflayergs(anet,
                   leaf_co2_ppress,
                   can_press,
                   veg_esat,
                   ceair,
                   stomatal_intercept_btran,
                   medlyn_slope,
                   gb_mol):
    term_gsmol = h2o_co2_stoma_diffuse_ratio * anet / (leaf_co2_ppress / can_press)
    vpd        = torch.clamp((veg_esat - ceair), min=50.0) * 0.001

    aquad = 1.0
    bquad = -(2.0 * (stomatal_intercept_btran + term_gsmol) + (medlyn_slope * term_gsmol) ** 2 / (gb_mol * vpd))
    cquad = stomatal_intercept_btran ** 2.0 + \
            (2.0 * stomatal_intercept_btran + term_gsmol * (1.0 - medlyn_slope ** 2.0 / vpd)) * term_gsmol

    # gs_mol computation updated here to avoid nan values when gs_mol quadratic function gives complex roots
    gs_mol = stomatal_intercept_btran.clone()
    mask   = (anet < 0.0) | (bquad * bquad < 4.0 * aquad * cquad)
    mask   = mask.type(torch.bool)
    gs_mol[~mask] = quadratic_max(aquad,bquad[~mask],cquad[~mask])
    return  gs_mol
#=======================================================================================================================
def GetResidual_ci(ci,
                   anet,
                   gs_mol,
                   can_co2_ppress,
                   can_press,
                   gb_mol,
                   # c3c4_path_index,
                   co2_ppress):

    f = ci - (can_co2_ppress - anet * can_press * (h2o_co2_bl_diffuse_ratio * gs_mol + h2o_co2_stoma_diffuse_ratio* gb_mol) / (gb_mol * gs_mol))
    ci_init = get_guess_ci(co2_ppress)# c3c4_path_index,
    f       = torch.where(anet < 0, ci - ci_init, f)
    return f
#=======================================================================================================================
def GetResidual_Tleaf(veg_tempk,  # kelvin
                      air_tempk,  # kelvin
                      APAR, # wm-2
                      rb, # (umol m-2 s-1)^-1
                      rs, # (umol m-2 s-1)^-1
                      veg_esat, # pa
                      ceair,# pa
                      can_press, # pa
                      ):
    lamda       = (2501000 - 2400 * (veg_tempk - 273.15)) * 1.4e-8
    theta_veg   = 1.0 / (1.0 + 0.076 / (Nlc * Cb))  # absorbed fraction,
    ARAD        = APAR + (APAR/theta_veg) * 0.6125
    f           =  veg_tempk -  (air_tempk + 1 / 38.4 * (ARAD - (lamda * (1/(rb+rs)) * torch.clamp(veg_esat - ceair, min = 0.0) / can_press)))
    return f
#=======================================================================================================================
def Updateveg_tempk(veg_tempk,  # kelvin
                    air_tempk,  # kelvin
                    APAR, # wm-2
                    rb, # (umol m-2 s-1)^-1
                    rs, # (umol m-2 s-1)^-1
                    veg_esat, # pa
                    ceair,# pa
                    can_press, # pa
                    ):
    # veg_temp = veg_temp.clone()
    # Niinemets, Ü. and Tenhunen, J. D.: ,Plant Cell Environ., 20, 845–866, 1997.
    #  %//the absorbed photon
    # APAR  = theta_veg * parsun_lsl
    # Molar heat content of water vapor (J umol^-1) ----> leaf temperature should be in degree celsius
    lamda = (2501000 - 2400 * (veg_tempk - 273.15)) * 1.4e-8
    # % assume the the PAR (400-700nm) is about 50%
    # of total radiation and absorption rate is 0.6125
    # for photosynthetic non-active radiation
    # add the non-photosynthetic active radiation
    theta_veg   = 1.0 / (1.0 + 0.076 / (Nlc * Cb))  # absorbed fraction,
    ARAD        = APAR + (APAR/theta_veg) * 0.6125
    # %tleafnew = tday + 1 / 38.4 * (ARAD - (lamda / rs * (ei - ea) / pressure));
    veg_tempnew =  (air_tempk + 1 / 38.4 * (ARAD - (lamda * (1/(rb+rs)) * torch.clamp(veg_esat - ceair, min = 0.0) / can_press)))
    # tleafnew = min(max(tleafnew, ConstList.Trange1), ConstList.Trange2);    # %constrain the physiological range
    return veg_tempnew
#=======================================================================================================================
def getPARlsl_In(kpft, PARin, cummulative_lai):
    return PARin * torch.exp(- kpft * cummulative_lai)
#=======================================================================================================================
def getqtop_layer( gs_mol,
                   cf,
                   c_area,
                   lai_lsl,
                   rb,
                   veg_esat,
                   ceair,
                   air_tempk,
                   can_press):
    # ! Convert gs_mol (umol /m**2/s) to gs (m/s) and then to rs (s/m)
    gs          = gs_mol / cf
    gstoma     = 1.0 / (torch.clamp(1.0 / gs, max=rsmax0)) # m/s
    rstoma_out = 1.0 / gstoma #s/m
    gsb_layer  = (1.0 / (rstoma_out + rb)) *  (lai_lsl * c_area)
    qsat       = (0.622 * veg_esat)/(can_press - 0.378 * veg_esat)
    qeair      = (0.622 * ceair)/(can_press - 0.378 * ceair)
    rho_atm    = (can_press - 0.378*ceair) /(Rda * air_tempk)
    # (veg_esat - ceair) * gsb_layer
    qtop       = rho_atm * (qsat - qeair) * gsb_layer
    return qtop
#=======================================================================================================================
def getnscaler(vcmax25top, leafn_vert_scaler_coeff1, leafn_vert_scaler_coeff2, cumulative_lai):
    kn = decay_coeff_vcmax(vcmax25top,
                           leafn_vert_scaler_coeff1,
                           leafn_vert_scaler_coeff2)

    nscaler = torch.exp(-kn * cumulative_lai)
    return nscaler












