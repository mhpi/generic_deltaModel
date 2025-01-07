import torch
from conf.Constants import denice, zsapric, pcalpha, pcbeta


def initsoilBCs(ipedof, sand, clay, om_frac, zsoi):
    watsat_col, bsw_col, sucsat_col, xksat = pedotransf(ipedof, sand, clay)

    om_watsat = torch.clamp(0.93 - 0.1 * (zsoi / zsapric), min=0.83)
    om_b      = torch.clamp(2.7 + 9.3 * (zsoi / zsapric), max=12.0)
    om_sucsat = torch.clamp(10.3 - 0.2 * (zsoi / zsapric), max=10.1)
    om_hksat  = torch.clamp(0.28 - 0.2799 * (zsoi / zsapric), min=0.0001)

    watsat_col = (1. - om_frac) * watsat_col + om_watsat * om_frac
    bsw_col    = (1. - om_frac) * (2.91 + 0.159 * clay) + om_frac * om_b
    sucsat_col = (1. - om_frac) * sucsat_col + om_sucsat * om_frac

    mask = om_frac > pcalpha
    mask = mask.bool()
    perc_norm = (1. - pcalpha) ** (-pcbeta)
    perc_frac = perc_norm * (om_frac - pcalpha) ** pcbeta
    perc_frac = torch.where(mask, perc_frac, 0.0)
    uncon_frac = (1. - om_frac) + (1. - perc_frac) * om_frac

    mask = om_frac < 1
    mask = mask.bool()
    uncon_hksat = uncon_frac / ((1. - om_frac) / xksat + ((1. - perc_frac) * om_frac) / om_hksat)
    uncon_hksat = torch.where(mask, uncon_hksat, 0.0)

    hksat_col = uncon_frac * uncon_hksat + (perc_frac * om_frac) * om_hksat

    return watsat_col, bsw_col, sucsat_col, hksat_col
def pedotransf_cosby1984_table4(sand, clay):
    # !
    # !DESCRIPTIONS
    # !compute hydraulic properties based on functions derived from Table 4 in cosby et al, 1984
    # use shr_kind_mod         , only : r8 => shr_kind_r8
    # implicit none
    # real(r8), intent(in) :: sand   !% sand
    # real(r8), intent(in) :: clay   !% clay
    # real(r8), intent(out):: watsat !v/v saturate moisture
    # real(r8), intent(out):: bsw    !b shape parameter
    # real(r8), intent(out):: sucsat !mm, soil matric potential
    # real(r8), intent(out):: xksat  !mm/s, saturated hydraulic conductivity

    # !Cosby et al. Table 4

    watsat = 0.505 - 0.00142 * sand - 0.00037 * clay
    bsw    = 3.10 + 0.157 * clay - 0.003 * sand
    sucsat = 10. * (10. ** (1.54 - 0.0095 * sand + 0.0063 * (100. - sand - clay)))
    xksat  = 0.0070556 * (10. ** (-0.60 + 0.0126 * sand - 0.0064 * clay))  # !mm/s now use table 4.

    return watsat, bsw, sucsat, xksat


# =======================================================================================================================
# =======================================================================================================================
def pedotransf_noilhan_lacarrere1995(sand, clay):
    # !
    # !DESCRIPTIONS
    # !compute hydraulic properties based on functions derived from Noilhan and Lacarrere, 1995
    #
    # use shr_kind_mod         , only : r8 => shr_kind_r8
    # implicit none
    # real(r8), intent(in) :: sand   !% sand
    # real(r8), intent(in) :: clay   !% clay
    # real(r8), intent(out):: watsat !v/v saturate moisture
    # real(r8), intent(out):: bsw    !b shape parameter
    # real(r8), intent(out):: sucsat !mm, soil matric potential
    # real(r8), intent(out):: xksat  !mm/s, saturated hydraulic conductivity

    # !Noilhan and Lacarrere, 1995

    watsat = -0.00108 * sand + 0.494305
    bsw = 0.137 * clay + 3.501
    sucsat = 10. ** (-0.0088 * sand + 2.85)
    xksat = 10. ** (-0.0582 * clay - 0.00091 * sand + 0.000529 * clay ** 2. - 0.0001203 * sand ** 2. - 1.38)
    return watsat, bsw, sucsat, xksat


# =======================================================================================================================
# =======================================================================================================================
def pedotransf_cosby1984_table5(sand, clay):
    # !
    # !DESCRIPTIONS
    # !compute hydraulic properties based on functions derived from Table 5 in cosby et al, 1984
    #
    # use shr_kind_mod         , only : r8 => shr_kind_r8
    # implicit none
    # real(r8), intent(in) :: sand   !% sand
    # real(r8), intent(in) :: clay   !% clay
    # real(r8), intent(out):: watsat !v/v saturate moisture
    # real(r8), intent(out):: bsw    !b shape parameter
    # real(r8), intent(out):: sucsat !mm, soil matric potential
    # real(r8), intent(out):: xksat  !mm/s, saturated hydraulic conductivity

    watsat = 0.489 - 0.00126 * sand
    bsw    = 2.91 + 0.159 * clay
    sucsat = 10. * (10. ** (1.88 - 0.0131 * sand))
    xksat  = 0.0070556 * (10. ** (-0.884 + 0.0153 * sand))  # ! mm/s, from table 5

    return watsat, bsw, sucsat, xksat


# =======================================================================================================================
# =======================================================================================================================
def pedotransf(ipedof, sand, clay):
    # !pedotransfer function to compute hydraulic properties of mineral soil
    # !based on input soil texture
    #
    # use shr_kind_mod         , only : r8 => shr_kind_r8
    # use abortutils    , only : endrun
    # implicit none
    # integer,  intent(in) :: ipedof !type of pedotransfer function, use the default pedotransfer function
    # real(r8), intent(in) :: sand   !% sand
    # real(r8), intent(in) :: clay   !% clay
    # real(r8), intent(out):: watsat !v/v saturate moisture
    # real(r8), intent(out):: bsw    !b shape parameter
    # real(r8), intent(out):: sucsat !mm, soil matric potential
    # real(r8), intent(out):: xksat  !mm/s, saturated hydraulic conductivity

    # character(len=32) :: subname = 'pedotransf'  ! subroutine name
    if ipedof == "cosby_1984_table4":
        watsat, bsw, sucsat, xksat = pedotransf_cosby1984_table4(sand, clay)
    elif ipedof == "noilhan_lacarrere_1995":
        watsat, bsw, sucsat, xksat = pedotransf_noilhan_lacarrere1995(sand, clay)
    elif ipedof == "cosby_1984_table5":
        watsat, bsw, sucsat, xksat = pedotransf_cosby1984_table5(sand, clay)
    else:
        raise ValueError(':: a pedotransfer function must be specified!')

    return watsat, bsw, sucsat, xksat


# =======================================================================================================================
# =======================================================================================================================
# # Example input values for sand and clay percentages
# sand_percentage = 40  # Example: 40%
# clay_percentage = 20  # Example: 20%
#
# # Calculate the pedotransfer function values
# watsat, bsw, sucsat, xksat = pedotransf_cosby1984_table5(sand_percentage, clay_percentage)
#
# print(watsat)
# print(bsw)
# print(sucsat)
# print(xksat)
# =======================================================================================================================
# =======================================================================================================================
def get_effporosity(watsat_col, h2osoi_ice,dz ):
    vol_ice = torch.min(watsat_col,h2osoi_ice/(dz*denice))
    eff_porosity = torch.clamp(watsat_col-vol_ice, min = 0.01)
    return eff_porosity