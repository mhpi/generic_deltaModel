import torch
from conf.Constants import  g_per_kg, kg_per_Megag, cm2_per_m2, fates_unset_r8, min_trim, branch_frac

def d2h_obrien(d, p1, p2, dbh_maxh):

    # real(r8),intent(in)    :: d        ! plant diameter [cm]
    # real(r8),intent(in)    :: p1       ! parameter a
    # real(r8),intent(in)    :: p2       ! parameter b
    # real(r8),intent(in)    :: p2       ! parameter b
    # real(r8),intent(in)    :: dbh_maxh ! dbh where max height occurs [cm]
    # real(r8),intent(out)   :: h        ! plant height [m]
    # real(r8),intent(out),optional   :: dhdd     ! change in height per diameter [m/cm]

    # !p1 = 0.64
    # !p2 = 0.37
    h = 10.0 ** (torch.log10(torch.minimum(d, dbh_maxh)) * p1 + p2)
    # if dhdd is not None:
    mask = d >= dbh_maxh
    # mask = mask.astype(bool)
    dhdd = torch.where(mask, 0.0, p1 * 10.0 ** p2 * d ** (p1 - 1.0))

    return h, dhdd

# ======================================================================================================
def d2h_poorter2006(d, p1, p2, p3, dbh_maxh):

    # ! "d2h_poorter2006"
    # ! "d to height via Poorter et al. 2006, these routines use natively
    # !  asymtotic functions (Weibull function)"
    # !
    # ! Poorter et al calculated height diameter allometries over a variety of
    # ! species in Bolivia, including those that could be classified in guilds
    # ! that were Partial-shade-tolerants, long-lived pioneers, shade-tolerants
    # ! and short-lived pioneers.  There results between height and diameter
    # ! found that small stature trees had less of a tendency to asymotote in
    # ! height and showed more linear relationships, and the largest stature
    # ! trees tended to show non-linear relationships that asymtote.
    # !
    # ! h = h_max*(1-exp(-a*d**b))
    # !
    # ! Poorter L, L Bongers and F Bongers.  Architecture of 54 moist-forest tree
    # ! species: traits, trade-offs, and functional groups.  Ecology 87(5), 2006.
    # !
    # ! =========================================================================

    # real(r8),intent(in)  :: d     ! plant diameter [cm]
    # real(r8),intent(in)     :: p1       ! parameter a = h_max
    # real(r8),intent(in)     :: p2       ! parameter b
    # real(r8),intent(in)     :: p3       ! parameter c
    # real(r8),intent(in)     :: dbh_maxh ! dbh at maximum height
    # real(r8),intent(out) :: h     ! plant height [m]
    # real(r8),intent(out),optional :: dhdd  ! change in height per diameter [m/cm]

    h = p1 * (1.0 - torch.exp(p2 * torch.minimum(d, dbh_maxh) ** p3))
    # !h = h_max - h_max (exp(a*d**b))
    # !f(x) = -h_max*exp(g(x))
    # !g(x) = a*d**b
    # !d/dx f(g(x) = f'(g(x))*g'(x) = -a1*exp(a2*d**a3) * a3*a2*d**(a3-1)

    # if dhdd is not None:
    mask = d >= dbh_maxh
    # mask = mask.bool()
    dhdd = torch.where(mask, 0.0, -p1 * torch.exp(p2 * d ** p3) * p3 * p2 * d ** (p3 - 1.0))

    return h, dhdd
# ======================================================================================================
def d2h_2pwr(d, p1, p2, dbh_maxh):

    # ! =========================================================================
    # ! "d2h_2pwr"
    # ! "d to height via 2 parameter power function"
    # ! where height h is related to diameter by a linear relationship on the log
    # ! transform where log(a) is the intercept and b is the slope parameter.
    # !
    # ! log(h) = log(a) + b*log(d)
    # ! h      = exp(log(a)) * exp(log(d))**b
    # ! h      = a*d**b
    # !
    # ! This functional form is used often in temperate allometries
    # ! Therefore, no base reference is cited.  Although, the reader is pointed
    # ! towards Dietze et al. 2008, King 1991, Ducey 2012 and many others for
    # ! reasonable parameters.  Note that this subroutine is intended only for
    # ! trees well below their maximum height, ie initialization.
    # !
    # ! =========================================================================
    # ! From King et al. 1990 at BCI for saplings
    # ! log(d) = a + b*log(h)
    # ! d = exp(a) * h**b
    # ! h = (1/exp(a)) * d**(1/b)
    # ! h = p1*d**p2  where p1 = 1/exp(a) = 1.07293  p2 = 1/b = 1.4925
    # ! d = (h/p1)**(1/p2)
    # ! For T. tuberculata (canopy tree) a = -0.0704, b = 0.67
    # ! =========================================================================
    #
    # ! args
    # ! =========================================================================
    # ! d: diameter at breast height
    # ! p1: the intercept parameter
    # !                       (however exponential of the fitted log trans)
    # ! p2: the slope parameter
    # ! return:
    # ! h: total tree height [m]
    # ! =========================================================================

    # real(r8),intent(in)     :: d        ! plant diameter [cm]
    # real(r8),intent(in)     :: p1       ! parameter a
    # real(r8),intent(in)     :: p2       ! parameter b
    # real(r8),intent(in)     :: dbh_maxh ! dbh where max height occurs [cm]
    # real(r8),intent(out)    :: h        ! plant height [m]
    # real(r8),intent(out),optional    :: dhdd     ! change in height per diameter [m/cm]

    h = p1 * torch.minimum(d, dbh_maxh) ** p2
    # if dhdd is not None:
    mask = d >= dbh_maxh
    # mask = mask.bool()
    dhdd = torch.where(mask, 0.0, (p2 * p1) * d ** (p2 - 1.0))

    return h, dhdd

# ======================================================================================================
def d2h_chave2014(d, p1, p2, p3, dbh_maxh):

    # ! "d2h_chave2014"
    # ! "d to height via Chave et al. 2014"
    #
    # ! This function calculates tree height based on tree diameter and the
    # ! environmental stress factor "E", as per Chave et al. 2015 GCB
    # ! As opposed to previous allometric models in ED, in this formulation
    # ! we do not impose a hard cap on tree height.  But, maximum_height
    # ! is an important parameter, but instead of imposing a hard limit, in
    # ! the new methodology, it will be used to trigger a change in carbon
    # ! balance accounting.  Such that a tree that hits its maximum height will
    # ! begin to route available NPP into seed and defense respiration.
    # !
    # ! The stress function is based on the geographic location of the site.  If
    # ! a user decides to use Chave2015 allometry, the E factor will be read in
    # ! from a global gridded dataset and assigned for each ED patch (note it
    # ! will be the same for each ED patch, but this distinction will help in
    # ! porting ED into different models (patches are pure ED).  It
    # ! assumes that the site is within the pan-tropics, and is a linear function
    # ! of climatic water deficit, temperature seasonality and precipitation
    # ! seasonality.  See equation 6b of Chave et al.
    # ! The relevant equation for height in this function is 6a of the same
    # ! manuscript, and is intended to pair with diameter to relate with
    # ! structural biomass as per equation 7 (in which H is implicit).
    # !
    # ! Chave et al. Improved allometric models to estimate the abovegroud
    # ! biomass of tropical trees.  Global Change Biology. V20, p3177-3190. 2015.
    # !
    # ! p1 =  0.893 - E
    # ! p2 =  0.76
    # ! p3 = -0.034
    # ! =========================================================================
    #
    # real(r8),intent(in)  :: d        ! plant diameter [cm]
    # real(r8),intent(in)  :: p1       ! parameter a
    # real(r8),intent(in)  :: p2       ! parameter b
    # real(r8),intent(in)  :: p3       ! parameter c
    # real(r8),intent(in)  :: dbh_maxh ! dbh at maximum height [cm]
    #
    # real(r8),intent(out) :: h     ! plant height [m]
    # real(r8),intent(out),optional :: dhdd  ! change in height per diameter [m/cm]
    # real(r8) :: p1e

    p1e = p1  # ! -eclim (assumed that p1 already has eclim removed)
    mask = d >= dbh_maxh
    # mask = mask.bool()
    h = torch.where(mask,
                 torch.exp(p1e + p2 * torch.log(dbh_maxh) + p3 * torch.log(dbh_maxh) ** 2.0),
                 torch.exp(p1e + p2 * torch.log(d) + p3 * torch.log(d) ** 2.0))
    # if dhdd is not None:
    dhdd = torch.where(mask, 0.0,
                    torch.exp(p1e) * (2.0 * p3 * d ** (p2 - 1.0 + p3 * torch.log(d)) * torch.log(d) +
                    p2 * d ** (p2 - 1.0 + p3 * torch.log(d))))

    return h, dhdd

# ======================================================================================================
def d2h_martcano(d, p1, p2, p3, dbh_maxh):
    #
    # ! =========================================================================
    # ! "d2h_martcano"
    # ! "d to height via 3 parameter Michaelis-Menten following work at BCI
    # ! by Martinez-Cano et al. 2016
    # !
    # ! h = (a*d**b)/(c+d**b)
    # !
    # ! h' = [(a*d**b)'(c+d**b) - (c+d**b)'(a*d**b)]/(c+d**b)**2
    # ! dhdd = h' = [(ba*d**(b-1))(c+d**b) - (b*d**(b-1))(a*d**b)]/(c+d**b)**2
    # !
    # ! args
    # ! =========================================================================
    # ! d: diameter at breast height
    # ! h: total tree height [m]
    # ! =========================================================================
    #
    # real(r8),intent(in)  :: d     ! plant diameter [cm]
    # real(r8),intent(in)  :: p1       ! parameter a
    # real(r8),intent(in)  :: p2       ! parameter b
    # real(r8),intent(in)  :: p3       ! parameter c
    # real(r8),intent(in)  :: dbh_maxh ! diameter at maximum height
    # real(r8),intent(out) :: h     ! plant height [m]
    # real(r8),intent(out),optional :: dhdd  ! change in height per diameter [m/cm]

    h = (p1 * torch.minimum(d, dbh_maxh) ** p2) / (p3 + torch.minimum(d, dbh_maxh) ** p2)
    # if dhdd is not None:
    mask = d >= dbh_maxh
    # mask = mask.bool()
    dhdd = torch.where(mask, 0.0, ((p2 * p1 * d ** (p2 - 1.)) * (p3 + d ** p2) -
                                   (p2 * d ** (p2 - 1.)) * (p1 * d ** p2)) /
                                   (p3 + d ** p2) ** 2.)

    return h, dhdd

# ======================================================================================================
def dh2bagw_salda(d, h, dhdd, p1, p2, p3, p4, wood_density, allom_agb_frac):
    # ! --------------------------------------------------------------------
    # ! Calculate stem biomass from height(m) dbh(cm) and wood density(g/cm3)
    # ! default params using allometry of J.G. Saldarriaga et al 1988 - Rio Negro
    # ! Journal of Ecology vol 76 p938-958
    # ! Saldarriaga 1988 provided calculations on total dead biomass
    # ! So here, we calculate total dead, and then remove the below-ground
    # ! fraction
    # ! --------------------------------------------------------------------
    #
    # real(r8),intent(in) :: d             ! plant diameter [cm]
    # real(r8),intent(in) :: h             ! plant height [m]
    # real(r8),intent(in) :: dhdd       ! change in height wrt diameter
    # real(r8),intent(in) :: p1         !    = 0.06896_r8
    # real(r8),intent(in) :: p2         !    = 0.572_r8
    # real(r8),intent(in) :: p3         !    = 1.94_r8
    # real(r8),intent(in) :: p4         !    = 0.931_r8
    # real(r8),intent(in) :: c2b            ! carbon 2 biomass ratio
    # real(r8),intent(in) :: wood_density
    # real(r8),intent(in) :: allom_agb_frac
    # real(r8),intent(out) :: bagw     ! plant biomass [kgC/indiv]
    # real(r8),intent(out),optional :: dbagwdd  ! change in agb per diameter [kgC/cm]
    #
    # real(r8) :: term1,term2,term3

    bagw = allom_agb_frac * p1 * (h ** p2) * (d ** p3) * (wood_density ** p4)

    # ! Add sapwood calculation to this?
    #
    # ! bagw     = a1 * h**a2 * d**a3 * r**a4
    # ! dbagw/dd = a1*r**a4 * d/dd (h**a2 * d**a3)
    # ! dbagw/dd = a1*r**a4 * [ d**a3 *d/dd(h**a2) + h**a2*d/dd(d**a3) ]
    # ! dbagw/dd = a1*r**a4 * [ d**a3 * a2*h**(a2-1)dh/dd + h**a2*a3*d**(a3-1)]
    #
    # ! From code
    # !   dbagw/dd =  a3 * a1 *(h**a2)*(d**(a3-1))* (r**a4) + a2*a1*(h**(a2-1))*(d**a3)*(r**a4)*dhdd
    # !   dbagw/dd =  a1*r**a4 * [ d**a3 * a2* h**(a2-1)*dhdd +  a3 * (h**a2)*(d**(a3-1)) ]

    # if dbagwdd is not None:

    term1 = allom_agb_frac * p1 * (wood_density ** p4)
    term2 = (h ** p2) * p3 * d ** (p3 - 1.0)
    term3 = p2 * h ** (p2 - 1.0) * (d ** p3) * dhdd
    dbagwdd = term1 * (term2 + term3)

    return bagw, dbagwdd

# ======================================================================================================
def d2bagw_2pwr(d, p1, p2, c2b):
    # ! =========================================================================
    # ! This function calculates tree above ground biomass according to 2
    # ! parameter power functions. (slope and intercepts of a log-log
    # ! diameter-agb fit:
    # !
    # ! These relationships are typical for temperate/boreal plants in North
    # ! America.  Parameters are available from Chojnacky 2014 and Jenkins 2003
    # !
    # ! Note that we are using an effective diameter here, as Chojnacky 2014
    # ! and Jenkins use diameter at the base of the plant for "woodland" species
    # ! The diameters should be converted prior to this routine if drc.
    # !
    # ! Input arguments:
    # ! diam: effective diameter (d or drc) in cm
    # ! FOR SPECIES THAT EXPECT DCM, THIS NEEDS TO BE PRE-CALCULATED!!!!
    # ! Output:
    # ! agb:   Total above ground biomass [kgC]
    # !
    # ! =========================================================================
    # ! Aceraceae, Betulaceae, Fagaceae and Salicaceae comprised nearly
    # ! three-fourths of the hardwood species (Table 3)
    # !
    # ! Fabaceae and Juglandaceae had specific gravities .0.60 and were
    # ! combined, as were Hippocastanaceae and Tilaceae with specific gravities
    # ! near 0.30. The remaining 9 families, which included mostly species with
    # ! specific gravity 0.45–0.55, were initially grouped to construct a general
    # ! hardwood taxon for those families having few published biomass equa-
    # ! tions however, 3 warranted separation, leaving 6 families for the general
    # ! taxon.
    # ! bagw = exp(b0 + b1*ln(diameter))/c2b
    # ! =========================================================================

    # real(r8),intent(in)  :: d       ! plant diameter [cm]
    # real(r8),intent(in)  :: p1      ! allometry parameter 1
    # real(r8),intent(in)  :: p2      ! allometry parameter 2
    # real(r8),intent(in)  :: c2b     ! carbon to biomass multiplier ~2
    # real(r8),intent(out) :: bagw    ! plant aboveground biomass [kg C]
    # real(r8),intent(out),optional :: dbagwdd  ! change in agb per diameter [kgC/cm]

    bagw = (p1 * d ** p2) / c2b
    # if dbagwdd is not None:
    dbagwdd = (p2 * p1 * d ** (p2 - 1.0)) / c2b

    return bagw, dbagwdd

# ======================================================================================================
def dh2bagw_chave2014(d, h, dhdd, p1, p2, wood_density, c2b):
    # ! =========================================================================
    # ! This function calculates tree structural biomass from tree diameter,
    # ! height and wood density.
    # !
    # ! Chave et al. Improved allometric models to estimate the abovegroud
    # ! biomass of tropical trees.  Global Change Biology. V20, p3177-3190. 2015.
    # !
    # ! Input arguments:
    # ! d: Diameter at breast height [cm]
    # ! rho:  wood specific gravity (dry wood mass per green volume)
    # ! height: total tree height [m]
    # ! a1: structural biomass allometry parameter 1 (intercept)
    # ! a2: structural biomass allometry parameter 2 (slope)
    # ! Output:
    # ! bagw:   Total above ground biomass [kgC]
    # !
    # ! Chave's Paper has p1 = 0.0673, p2 = 0.976
    # !
    #
    # ! =========================================================================

    # real(r8),intent(in)  :: d       ! plant diameter [cm]
    # real(r8),intent(in)  :: h       ! plant height [m]
    # real(r8),intent(in)  :: dhdd    ! change in height wrt diameter
    # real(r8),intent(in)  :: p1  ! allometry parameter 1
    # real(r8),intent(in)  :: p2  ! allometry parameter 2
    # real(r8),intent(in)  :: wood_density
    # real(r8),intent(in)  :: c2b
    # real(r8),intent(out) :: bagw     ! plant aboveground biomass [kgC]
    # real(r8),intent(out),optional :: dbagwdd  ! change in agb per diameter [kgC/cm]
    #
    # real(r8) :: dbagwdd1,dbagwdd2,dbagwdd3

    bagw = (p1 * (wood_density * d ** 2.0 * h) ** p2) / c2b
    # if dbagwdd is not None:
    # ! Need the the derivative of height to diameter to
    # ! solve the derivative of agb with height
    dbagwdd1 = (p1 * wood_density ** p2) / c2b
    dbagwdd2 = p2 * d ** (2.0 * p2) * h ** (p2 - 1.0) * dhdd
    dbagwdd3 = h ** p2 * 2.0 * p2 * d ** (2.0 * p2 - 1.0)
    dbagwdd = dbagwdd1 * (dbagwdd2 + dbagwdd3)

    return bagw, dbagwdd
# ======================================================================================================
def GetCrownReduction(crowndamage):
    # !------------------------------------------------------------------
    # ! This subroutine takes the crown damage class of a cohort (integer)
    # ! and returns the fraction of the crown that is lost.
    # !-------------------------------------------------------------------

    # integer(i4), intent(in)   :: crowndamage        ! crown damage class of the cohort
    # real(r8),    intent(out)  :: crown_reduction    ! fraction of crown lost from damage

    # This is manually changed
    crown_reduction = 0.0  # ED_val_history_damage_bin_edges(crowndamage)/100.0

    return crown_reduction
# ======================================================================================================
def d2blmax_salda(d, p1, p2, p3, rho, dbh_maxh):
    # real(r8),intent(in)    :: d         ! plant diameter [cm]
    # real(r8),intent(in)    :: p1
    # real(r8),intent(in)    :: p2
    # real(r8),intent(in)    :: p3
    # real(r8),intent(in)    :: rho       ! plant's wood specific gravity
    # real(r8),intent(in)    :: dbh_maxh  ! dbh at which max height occurs
    # real(r8),intent(in)    :: c2b       ! c to biomass multiplier (~2.0)
    #
    # real(r8),intent(out)   :: blmax     ! plant leaf biomass [kg]
    # real(r8),intent(out),optional   :: dblmaxdd  ! change leaf bio per diam [kgC/cm]
    mask = d < dbh_maxh
    # mask = mask.bool()
    blmax = torch.where(mask, p1 * d ** p2 * rho ** p3, p1 * dbh_maxh ** p2 * rho ** p3)

    # if dblmaxdd is not None:
    dblmaxdd = torch.where(mask, p1 * p2 * d ** (p2 - 1.0) * rho ** p3, 0.0)

    return blmax, dblmaxdd
# ======================================================================================================

def d2blmax_2pwr(d, p1, p2, c2b):
    # ! ======================================================================
    # ! This is a power function for leaf biomass from plant diameter.
    # ! ======================================================================
    #
    # ! p1 and p2 represent the parameters that govern total beaf dry biomass,
    # ! and the output argument blmax is the leaf carbon only
    #
    # real(r8),intent(in)  :: d         ! plant diameter [cm]
    # real(r8),intent(in)  :: p1        ! parameter 1  (slope)
    # real(r8),intent(in)  :: p2        ! parameter 2  (curvature, exponent)
    # real(r8),intent(in)  :: c2b       ! carbon to biomass multiplier (~2)
    #
    # real(r8),intent(out) :: blmax     ! plant leaf biomass [kgC]
    # real(r8),intent(out),optional :: dblmaxdd  ! change leaf bio per diameter [kgC/cm]

    blmax = (p1 * d ** p2) / c2b
    # if dblmaxdd is not None:
    dblmaxdd = p1 * p2 * d ** (p2 - 1.0) / c2b

    return blmax, dblmaxdd

# ======================================================================================================

def dh2blmax_2pwr(d, p1, p2, dbh_maxh, c2b):
    # ! -------------------------------------------------------------------------
    # ! This formulation is very similar to d2blmax_2pwr
    # ! The difference is that for very large trees that have reached peak
    # ! height, we cap leaf biomass.
    # ! --------------------------------------------------------------------------
    #
    # real(r8),intent(in)  :: d         ! plant diameter [cm]
    # real(r8),intent(in)  :: p1        ! parameter 1 (slope)
    # real(r8),intent(in)  :: p2        ! parameter 2 (curvature, exponent)
    # real(r8),intent(in)  :: c2b       ! carbon 2 biomass multiplier
    # real(r8),intent(in)  :: dbh_maxh  ! dbh at maximum height
    #
    # real(r8),intent(out) :: blmax     ! plant leaf biomass [kgC]
    # real(r8),intent(out),optional :: dblmaxdd  ! change leaf bio per diameter [kgC/cm]
    #
    # ! reproduce Saldarriaga:
    # ! blmax = p1 * dbh_maxh**p2 * rho**p3
    # !       = 0.07 * dbh_maxh**p2 * 0.7*0.55 = (p1 + p2* dbh**p3) / c2b
    # !       p1 = 0
    # !       p2 = (0.07 * 0.7^0.55)*2 = 0.11506201034678605

    # not sure if I should use torch.min or torch.clamp
    blmax = (p1 * torch.minimum(d, dbh_maxh) ** p2) / c2b

    # ! If this plant has reached its height cap, then it is not
    # ! adding leaf mass.  In this case, dhdd = 0
    # if dblmaxdd is not None:
    mask = d >= dbh_maxh
    # mask = mask.bool()
    dblmaxdd = torch.where(mask, 0.0, p1 * p2 * d ** (p2 - 1.0) / c2b)

    return blmax, dblmaxdd

# ======================================================================================================
def bsap_ltarg_slatop(prt_params, h, dhdd, bleaf, dbleafdd):
    # ! -------------------------------------------------------------------------
    # ! Calculate sapwood carbon based on its leaf area per sapwood area
    # ! proportionality with the plant's target leaf area.
    # ! of plant size, see Calvo-Alvarado and Bradley Christoferson.
    # !
    # ! Important note 1: This is above and below-ground sapwood
    # ! Important note 2: Since we need continuous calculation of
    # !                   sapwood dependent on plant size, we cannot
    # !                   use actual leaf area (which is canopy dependent).
    # !                   So, this method estimates a leaf area that is
    # !                   based only on the specific leaf area (SLA) of
    # !                   the canopy top.
    # !
    # ! -------------------------------------------------------------------------

    # real(r8),intent(in)    :: d         ! plant diameter [cm]
    # real(r8),intent(in)    :: h         ! plant height [m]
    # real(r8),intent(in)    :: dhdd      ! change in height per diameter [m/cm]
    # real(r8),intent(in)    :: bleaf     ! plant leaf target biomass [kgC]
    # real(r8),intent(in)    :: dbleafdd  ! change in blmax per diam [kgC/cm]
    # integer(i4),intent(in) :: ipft      ! PFT index
    # real(r8),intent(out)   :: sapw_area ! area of sapwood crosssection at
    #                                     ! reference height [m2]
    # real(r8),intent(out)   :: bsap      ! plant leaf biomass [kgC]
    # real(r8),intent(out),optional :: dbsapdd   ! change leaf bio per diameter [kgC/cm]
    #
    # real(r8)               :: la_per_sa  ! effective leaf area per sapwood area
    #                                      ! [m2/cm]
    # real(r8)               :: term1      ! complex term for solving derivative
    # real(r8)               :: dterm1_dh  ! deriv of term1 wrt height
    # real(r8)               :: dterm1_dd  ! deriv of term1 wrt diameter
    # real(r8)               :: hbl2bsap   ! sapwood biomass per lineal height
    la_per_sa_int   = prt_params['la_per_sa_int']
    la_per_sa_slp   = prt_params['la_per_sa_slp']
    slatop          = prt_params['slatop']
    wood_density    = prt_params['wood_density']
    c2b             = prt_params['c2b']
    agb_fraction    = prt_params['agb_frac']

    # ! Calculate sapwood biomass per linear height and kgC of leaf [m-1]
    # ! Units:
    # ! Note: wood_density is in units of specific gravity, which is also
    # !       Mg / m3  (megagrams, ie 1000 kg / m3)
    # ! 1 /la_per_sa * slatop*     gtokg    *   cm2tom2     / c2b     * mg2kg  * dens
    # ! [cm2/m2]     * [m2/gC]*[1000gC/1kgC]*[1m2/10000cm2] /[kg/kgC]*[kg/Mg]*[Mg/m3]
    # !        ->[cm2/gC]
    # !                  ->[cm2/kgC]
    # !                                ->[m2/kgC]
    # !                                             ->[m2/kg]
    # !                                                       ->[m2/Mg]
    # !                                                                  ->[/m]
    # ! ------------------------------------------------------------------------
    #
    # ! This is a term that combines unit conversion and specific leaf
    # ! area.  This term does not contain the proportionality
    # ! between leaf area and sapwood cross-section. This is
    # ! because their may be a height dependency, and will effect the
    # ! derivative wrt diameter.

    hbl2bsap = slatop * g_per_kg * wood_density * kg_per_Megag / (c2b * cm2_per_m2)

    # ! Calculate area. Note that no c2b conversion here, because it is
    # ! wood density that is in biomass units, SLA is in units [m2/gC.
    # ! [m2]    = [m2/gC] * [kgC] * [gC/kgC] / ( [m2/cm2] * [cm2/m2])
    la_per_sa = la_per_sa_int + h * la_per_sa_slp
    sapw_area = slatop * bleaf * g_per_kg / (la_per_sa * cm2_per_m2)
    # I left it as it is since not used in the rest of the function and not an output so useless otherwise
    # need to change the code: if sapwood area appears anywhere after this line. We need to use this only if
    # sapwood area is nan

    # ! Note the total depth of the plant is approximated by the
    # ! above ground fraction. This fraction is actually associated
    # ! with biomass, but we use it here as well to help us assess
    # ! how much sapwood is above and below ground.
    # ! total_depth * agb_fraction = height
    #
    # ! Integrate the mass per leaf biomass per depth of the plant
    # ! Include above and below ground components
    # ! [kgC] = [kgC/kgC/m]   * [kgC]     * [m]
    # ! ------------------------------------------------------------------------

    # !      bsap =  hbl2bsap/(la_per_sa_int + h*la_per_sa_slp) * (h/agb_fraction) * bleaf
    #
    #       ! "term1" combines two height dependent functions. The numerator is
    #       ! how sapwood volume scales in the vertical direction.  The denominator
    #       ! is the leaf_area per sapwood area ratio [m2/cm2], which is height dependent
    #       ! (for non-zero slope parameters)

    term1 = h / (la_per_sa_int + h * la_per_sa_slp)
    bsap  = (hbl2bsap / agb_fraction) * term1 * bleaf

    # ! dbldmaxdd is deriv of blmax wrt dbh (use directives to check oop)
    # ! dhdd is deriv of height wrt dbh (use directives to check oop)
    # if dbsapdd is not None:

    dterm1_dh = la_per_sa_int / (la_per_sa_int + la_per_sa_slp * h) ** 2.0
    dterm1_dd = dterm1_dh * dhdd
    dbsapdd   = hbl2bsap / agb_fraction * (bleaf * dterm1_dd + term1 * dbleafdd)

    return sapw_area, bsap, dbsapdd

# ======================================================================================================
#   ! Fine root biomass allometry wrapper
def blmax_allom(d, prt_params):
    # d = sites.plant_diam
    dbh_maxh = prt_params['dbh_maxh']
    rho      = prt_params['wood_density']
    c2b      = prt_params['c2b']
    p1       = prt_params['d2bl1']
    p2       = prt_params['d2bl2']
    p3       = prt_params['d2bl3']
    allom_lmode  = prt_params['allom_lmode']

    blmax    = torch.full_like(d, fates_unset_r8)
    dblmaxdd = torch.full_like(d, fates_unset_r8)

    for mode in range(1, 4):
        mask = (allom_lmode == mode).bool()#.astype(bool)#

        if mode == 1:
            blmax__, dblmaxdd__ = d2blmax_salda(d, p1, p2, p3, rho, dbh_maxh)
        elif mode ==2:
            blmax__, dblmaxdd__ = d2blmax_2pwr(d, p1, p2, c2b)
        else:
            blmax__, dblmaxdd__ = dh2blmax_2pwr(d, p1, p2, dbh_maxh, c2b)

        blmax    = torch.where(mask, blmax__, blmax)
        dblmaxdd = torch.where(mask, dblmaxdd__, dblmaxdd)

    return blmax, dblmaxdd
# ======================================================================================================
def bleaf(d, prt_params,crowndamage,canopy_trim,elongf_leaf):

    # ! -------------------------------------------------------------------------
    # ! This subroutine calculates the actual target bleaf
    # ! based on trimming. Because trimming
    # ! is not allometry and rather an emergent property,
    # ! this routine is not name-spaced with allom_
    # ! -------------------------------------------------------------------------
    #
    # use DamageMainMod      , only : GetCrownReduction
    #
    # real(r8),intent(in)    :: d             ! plant diameter [cm]
    # integer(i4),intent(in) :: ipft          ! PFT index
    # integer(i4),intent(in) :: crowndamage   ! crown damage class [1: undamaged, >1: damaged]
    # real(r8),intent(in)    :: canopy_trim   ! trimming function
    # real(r8),intent(in)    :: elongf_leaf   ! Leaf elongation factor (phenology)
    # real(r8),intent(out)   :: bl            ! plant leaf biomass [kgC]
    # real(r8),intent(out),optional :: dbldd  ! change leaf bio per diameter [kgC/cm]
    #
    # real(r8) :: blmax
    # real(r8) :: dblmaxdd
    # real(r8) :: crown_reduction

    blmax, dblmaxdd = blmax_allom(d, prt_params)

    # ! -------------------------------------------------------------------------
    # ! Adjust for canopies that have become so deep that their bottom layer is
    # ! not producing any carbon...
    # ! nb this will change the allometry and the effects of this remain untested
    # ! RF. April 2014
    # ! -------------------------------------------------------------------------

    bl = blmax * canopy_trim

    # if dbldd is not None:
    dbldd = dblmaxdd * canopy_trim

    # ! Potentially reduce leaf biomass based on crown damage (crown_reduction) and/or
    # ! phenology (elongf_leaf).
    if ( crowndamage > 1 ):

        crown_reduction = GetCrownReduction(crowndamage)
        bl = elongf_leaf * bl * (1.0 - crown_reduction)
        # if dbldd is not None:
        dbldd = elongf_leaf * dblmaxdd * canopy_trim * (1.0 - crown_reduction)
    else:
        bl = elongf_leaf * bl
        # if dbldd is not None:
        dbldd = elongf_leaf * dbldd

    return bl, dbldd

# ======================================================================================================
def bfineroot(d,prt_params, canopy_trim,l2fr,elongf_fnrt):

    # ! -------------------------------------------------------------------------
    # ! This subroutine calculates the actual target fineroot biomass
    # ! based on functions that may or may not have prognostic properties.
    # ! -------------------------------------------------------------------------
    #
    # real(r8),intent(in)    :: d             ! plant diameter [cm]
    # integer(i4),intent(in) :: ipft          ! PFT index
    # real(r8),intent(in)    :: canopy_trim   ! trimming function
    # real(r8),intent(in)    :: l2fr          ! leaf to fineroot scaler
    #                                         ! this is either a PFT parameter
    #                                         ! constant (when no nutrient model)
    #                                         ! or dynamic (with nutrient model)
    # real(r8),intent(in)    :: elongf_fnrt   ! Elongation factor for fine roots
    # real(r8),intent(out)   :: bfr           ! fine root biomass [kgC]
    # real(r8),intent(out),optional :: dbfrdd ! change leaf bio per diameter [kgC/cm]
    #
    # real(r8) :: blmax      ! maximum leaf biomss per allometry
    # real(r8) :: dblmaxdd
    # real(r8) :: bfrmax
    # real(r8) :: dbfrmaxdd
    # real(r8) :: slascaler

    allom_fmode     = prt_params['allom_fmode']
    blmax, dblmaxdd = blmax_allom(d, prt_params)

    "constant proportionality with TRIMMED target bleaf"
    mask1 = allom_fmode == 1; mask1 = mask1.bool()
    "constant proportionality with UNTRIMMED target bleaf"
    mask2 = allom_fmode == 2; mask2 = mask2.bool()

    bfr = torch.where(mask1, blmax * l2fr * canopy_trim,
                   torch.where(mask2, blmax * l2fr, fates_unset_r8))
    # if dbfrdd is not None:
    dbfrdd = torch.where(mask1,dblmaxdd * l2fr * canopy_trim,
                      torch.where(mask2,dblmaxdd * l2fr, fates_unset_r8 ) )

    # ! Reduce fine-root biomass due to phenology.
    bfr = elongf_fnrt * bfr
    # if dbfrdd is not None:
    dbfrdd = elongf_fnrt * dbfrdd

    return bfr,dbfrdd
# ======================================================================================================
def bbgw_const(prt_params, bagw,dbagwdd):

    # real(r8),intent(in)    :: d         ! plant diameter [cm]
    # real(r8),intent(in)    :: bagw       ! above ground biomass [kg]
    # real(r8),intent(in)    :: dbagwdd    ! change in agb per diameter [kg/cm]
    # integer(i4),intent(in) :: ipft      ! PFT index
    # real(r8),intent(out)   :: bbgw       ! coarse root biomass [kg]
    # real(r8),intent(out),optional :: dbbgwdd    ! change croot bio per diam [kg/cm]

    agb_fraction = prt_params['agb_frac']

    bbgw = (1.0/agb_fraction-1.0)*bagw

    # ! Derivative
    # ! dbbgw/dd = dbbgw/dbagw * dbagw/dd
    #  dbbgwdd is not None:
    dbbgwdd = (1.0/agb_fraction-1.0)*dbagwdd

    return bbgw, dbbgwdd

# ======================================================================================================
def bdead_allom(prt_params,
                bagw,bbgw,bsap,
                dbagwdd = None,
                dbbgwdd = None,
                dbsapdd = None,
                ):

    # real(r8),intent(in)  :: bagw      ! biomass above ground wood (agb) [kgC]
    # real(r8),intent(in)  :: bbgw       ! biomass below ground (bgb) [kgC]
    # real(r8),intent(in)  :: bsap      ! sapwood biomass [kgC]
    # integer(i4),intent(in) :: ipft    ! PFT index
    # real(r8),intent(out) :: bdead     ! dead biomass (heartw/struct) [kgC]
    #
    # real(r8),intent(in),optional  :: dbagwdd    ! change in agb per d [kgC/cm]
    # real(r8),intent(in),optional  :: dbbgwdd    ! change in bbgw per d [kgC/cm]
    # real(r8),intent(in),optional  :: dbsapdd   ! change in bsap per d [kgC/cm]
    # real(r8),intent(out),optional :: dbdeaddd  ! change in bdead per d [kgC/cm]

    # ! bdead is diagnosed as the mass balance from all other pools
    # ! and therefore, no options are necessary
    # !
    # ! Assumption: We assume that the leaf biomass component of AGB is negligable
    # ! and do not assume it is part of the AGB measurement, nor are fineroots part of the
    # ! bbgw. Therefore, it is not removed from AGB and BBGW in the calculation of dead mass.

    allom_agb_frac = prt_params['agb_frac']
    allom_amode    = prt_params['allom_amode']

    agb_fraction = allom_agb_frac
    # ! Saldariagga mass allometry originally calculated bdead directly.
    # ! we assume proportionality between bdead and bagw
    mask1 = allom_amode == 1; mask1 = mask1.bool()
    mask23= (allom_amode == 2) | (allom_amode == 3); mask23 = mask23.bool()

    bdead = torch.where(mask1,bagw/agb_fraction,
                     torch.where(mask23, bagw + bbgw - bsap, fates_unset_r8) )

    if dbagwdd is not None and dbbgwdd is not None and dbsapdd is not None:
        dbdeaddd = torch.where(mask1, dbagwdd/agb_fraction,
                           torch.where(mask23, dbagwdd+dbbgwdd-dbsapdd, fates_unset_r8))
    else:
        dbdeaddd = None

    return bdead, dbdeaddd

# ======================================================================================================
def h_allom(d, prt_params):
    # d = sites.plant_diam

    # ! Arguments
    # real(r8),intent(in)     :: d     ! plant diameter [cm]
    # integer(i4),intent(in)  :: ipft  ! PFT index
    # real(r8),intent(out)    :: h     ! plant height [m]
    # real(r8),intent(out),optional :: dhdd  ! change in height per diameter [m/cm]

    dbh_maxh    = prt_params['dbh_maxh']
    p1          = prt_params['d2h1']
    p2          = prt_params['d2h2']
    p3          = prt_params['d2h3']
    allom_hmode = prt_params['allom_hmode']

    h    = torch.full_like(d, fates_unset_r8)
    dhdd = torch.full_like(d, fates_unset_r8)

    for mode in range(1, 6):
        mask = (allom_hmode == mode).bool()#.astype(bool)#

        if mode == 1:# "obrien"
            h__, dhdd__ = d2h_obrien(d,p1,p2,dbh_maxh)
        elif mode ==2:# "poorter06"
            h__, dhdd__ = d2h_poorter2006(d,p1,p2,p3,dbh_maxh)
        elif mode ==3:# "2parameter power function h=a*d^b "
            h__, dhdd__ = d2h_2pwr(d, p1, p2, dbh_maxh)
        elif mode ==4:# "chave14"
            h__, dhdd__ = d2h_chave2014(d, p1, p2, p3, dbh_maxh)
        else:# " Martinez-Cano"
            h__, dhdd__ = d2h_martcano(d,p1,p2,p3,dbh_maxh)

        h    = torch.where(mask, h__,h)
        dhdd = torch.where(mask, dhdd__, dhdd)

    return h, dhdd

# ======================================================================================================
def bagw_allom(d, h, prt_params, crowndamage, elongf_stem):
    # Updated 16th May sunch that plant height is an input and not computed inside the function

    # d = sites.plant_diam
    # use DamageMainMod, only : GetCrownReduction
    # use FatesParameterDerivedMod, only : param_derived

    # real(r8),intent(in)    :: d       ! plant diameter [cm]
    # integer(i4),intent(in) :: ipft    ! PFT index
    # integer(i4),intent(in) :: crowndamage ! crowndamage [1: undamaged, >1: damaged]
    # real(r8),intent(in)    :: elongf_stem ! Stem elongation factor
    # real(r8),intent(out)   :: bagw    ! biomass above ground woody tissues
    # real(r8),intent(out),optional :: dbagwdd  ! change in agbw per diameter [kgC/cm]
    #
    # real(r8)               :: h       ! height
    # real(r8)               :: dhdd    ! change in height wrt d
    # real(r8)               :: crown_reduction  ! crown reduction from damage
    # real(r8)               :: branch_frac ! fraction of aboveground woody biomass in branches

    p1          = prt_params['agb1']
    p2          = prt_params['agb2']
    p3          = prt_params['agb3']
    p4          = prt_params['agb4']
    wood_density= prt_params['wood_density']
    c2b         = prt_params['c2b']
    agb_frac    = prt_params['agb_frac']
    allom_amode = prt_params['allom_amode']


    #branch_frac = # param_derived%branch_frac(ipft)
    # branch_frac = sum(prt_params['fates_frag_cwd_frac'][:3])

    bagw    = torch.full_like(d, fates_unset_r8)
    dbagwdd = torch.full_like(d, fates_unset_r8)

    _, dhdd = h_allom(d, prt_params)

    for mode in range(1,4):
        mask = (allom_amode == mode).bool()
        if mode ==1: # "salda"
            bagw__, dbagwdd__  = dh2bagw_salda(d, h, dhdd, p1, p2, p3, p4, wood_density, agb_frac)
        elif mode ==2:# "2par_pwr"
            bagw__, dbagwdd__  = d2bagw_2pwr(d,p1,p2,c2b)
        else:# "chave14"
            bagw__, dbagwdd__  = dh2bagw_chave2014(d,h,dhdd,p1,p2,wood_density,c2b)

        bagw    = torch.where(mask, bagw__, bagw)
        dbagwdd = torch.where(mask, dbagwdd__, dbagwdd)


    # ! Potentially reduce AGB based on crown damage (crown_reduction) and/or
    # ! phenology (elongf_stem).
    if(crowndamage > 1) :
        crown_reduction = GetCrownReduction(crowndamage)
        bagw = elongf_stem * ( bagw - (bagw * branch_frac * crown_reduction) )
        # if dbagwdd is not None:
        dbagwdd = elongf_stem * ( dbagwdd - (dbagwdd * branch_frac * crown_reduction) )
    else:
        bagw = elongf_stem * bagw
        # if dbagwdd is not None:
        dbagwdd = elongf_stem * dbagwdd

    return bagw, dbagwdd
# ======================================================================================================
def bbgw_allom(d, h, prt_params, elongf_stem):
    # Updated 16th May, plant height h is an input

    # real(r8),intent(in)           :: d           ! plant diameter [cm]
    # integer(i4),intent(in)        :: ipft        ! PFT index
    # real(r8),intent(in)           :: elongf_stem ! Elongation factor for stems (phenology)
    # real(r8),intent(out)          :: bbgw        ! below ground woody biomass [kgC]
    # real(r8),intent(out),optional :: dbbgwdd     ! change bbgw  per diam [kgC/cm]
    #
    # real(r8)    :: bagw       ! above ground biomass [kgC]

    allom_cmode = prt_params['allom_cmode']

    mask = allom_cmode == 1; mask = mask.bool()

    # ! bbgw not affected by damage so use target allometry no damage. But note that bbgw
    # ! is affected by stem phenology (typically applied only to grasses). We do not need
    # ! to account for stem phenology in bbgw_const because bbgw will be proportional to
    # ! bagw, and bagw is downscaled due to stem phenology.
    bagw,dbagwdd = bagw_allom(d, h, prt_params, crowndamage = 1, elongf_stem = elongf_stem)
    bagw         = torch.where(mask, bagw, fates_unset_r8)
    dbagwdd      = torch.where(mask, dbagwdd, fates_unset_r8)

    bbgw, dbbgwdd= bbgw_const(prt_params,bagw,dbagwdd)

    return bbgw, dbbgwdd
# ======================================================================================================
def bsap_allom(d, h, prt_params, crowndamage,canopy_trim,elongf_stem):
    # 16th May updated so that plant height is external input and not computed inside the function


    # use DamageMainMod , only : GetCrownReduction
    # use FatesParameterDerivedMod, only : param_derived
    #
    # real(r8),intent(in)           :: d           ! plant diameter [cm]
    # integer(i4),intent(in)        :: ipft        ! PFT index
    # integer(i4),intent(in)        :: crowndamage ! Crown damage class [1: undamaged, >1: damaged]
    # real(r8),intent(in)           :: canopy_trim
    # real(r8),intent(in)           :: elongf_stem ! Elongation factor for stems (phenology)
    # real(r8),intent(out)          :: sapw_area   ! cross section area of
    #                                              ! plant sapwood at reference [m2]
    # real(r8),intent(out)          :: bsap        ! sapwood biomass [kgC]
    # real(r8),intent(out),optional :: dbsapdd     ! change in sapwood biomass
    #                                              ! per d [kgC/cm]
    #
    # real(r8) :: h         ! Plant height [m]
    # real(r8) :: dhdd
    # real(r8) :: bl
    # real(r8) :: dbldd
    # real(r8) :: bbgw
    # real(r8) :: dbbgwdd
    # real(r8) :: bagw
    # real(r8) :: dbagwdd
    # real(r8) :: bsap_cap  ! cap sapwood so that it is no larger
    #                       ! than some specified proportion of woody biomass
    #                       ! should not trip, and only in small plants
    #
    # real(r8) :: crown_reduction  ! amount that crown is damage by
    # real(r8) :: agb_frac         ! aboveground biomass fraction
    # real(r8) :: branch_frac      ! fraction of aboveground woody biomass in branches
    #
    # ! Constrain sapwood so that its above ground portion be no larger than
    # ! X% of total woody/fibrous (ie non leaf/fineroot) tissues
    max_frac = 0.95
    #branch_frac = sum(prt_params['fates_frag_cwd_frac'][:3])


    agb_frac    = prt_params['agb_frac']
    allom_smode = prt_params['allom_smode']
    #branch_frac =  param_derived%branch_frac(ipft)

    _, dhdd   = h_allom(d, prt_params)
    bl, dbldd = bleaf(d= d, prt_params=prt_params, crowndamage = 1, canopy_trim = canopy_trim, elongf_leaf = 1)

    # I left it as it is since not used in the rest of the function and not an output so useless otherwise
    # need to change the code if sapwood area appears anywhere after this line, We need to use this only if
    # sapwood area is nan
    sapw_area,bsap,dbsapdd = bsap_ltarg_slatop(prt_params,h,dhdd,bl,dbldd)

    #! linearly related to leaf area based on target leaf biomass
    #! and slatop (no provisions for slamax)
    mask = allom_smode == 1; mask = mask.bool()

    # ! ---------------------------------------------------------------------
    # ! Currently only one sapwood allometry model. the slope
    # ! of the la:sa to diameter line is zero.
    # ! ---------------------------------------------------------------------
    # !  We assume fully flushed leaves, so sapwood biomass is independent of leaf phenology
    # ! (but could be modulated by stem phenology).

    # ! if trees are damaged reduce bsap by percent crown loss *
    # ! fraction of biomass that would be in branches (pft specific)
    if(crowndamage > 1):
        crown_reduction =  GetCrownReduction(crowndamage)
        bsap = elongf_stem * ( bsap - (bsap * agb_frac *  branch_frac * crown_reduction) )
        # if dbsapdd is not None:
        dbsapdd = elongf_stem * ( dbsapdd - (dbsapdd * agb_frac * branch_frac * crown_reduction) )
    else:
        bsap = elongf_stem * bsap
        # if dbsapdd is not None:
        dbsapdd = elongf_stem * dbsapdd

    # ! Perform a capping/check on total woody biomass
    bagw,dbagwdd = bagw_allom(d, h, prt_params, crowndamage, elongf_stem)
    bbgw,dbbgwdd = bbgw_allom(d, h, prt_params, elongf_stem)

    # ! Force sapwood to be less than a maximum fraction of total biomass
    # ! We omit the sapwood area from this calculation
    # ! (this comes into play typically in very small plants)

    bsap_cap = max_frac*(bagw+bbgw)
    bsap     = torch.where(mask, bsap, fates_unset_r8)

    mask    = mask & (bsap>bsap_cap); mask = mask.bool()
    bsap    = torch.where(mask,bsap_cap, bsap )
    dbsapdd = torch.where(mask, max_frac * (dbagwdd + dbbgwdd), dbsapdd)


    return bsap, dbsapdd
# ======================================================================================================
def  carea_2pwr(dbh,spread,d2bl_p2,d2bl_ediff,d2ca_min, d2ca_max,crowndamage,c_area = None, inverse = False):

    # ! ============================================================================
    # ! Calculate area of ground covered by entire cohort. (m2)
    # ! Function of DBH (cm) canopy spread (m/cm) and number of individuals.
    # ! ============================================================================

    # real(r8),intent(inout) :: dbh      ! diameter at breast (refernce) height [cm]
    # real(r8),intent(in) :: spread      ! site level relative spread score [0-1]
    # real(r8),intent(in) :: d2bl_p2     ! parameter 2 in the diameter->bleaf allometry (exponent)
    # real(r8),intent(in) :: d2bl_ediff  ! area difference factor in the diameter-bleaf allometry (exponent)
    # real(r8),intent(in) :: d2ca_min    ! minimum diameter to crown area scaling factor
    # real(r8),intent(in) :: d2ca_max    ! maximum diameter to crown area scaling factor
    # integer,intent(in)  :: crowndamage ! crowndamage class [1: undamaged, >1: damaged]
    # real(r8),intent(inout) :: c_area   ! crown area for one plant [m2]
    # logical,intent(in)  :: inverse     ! if true, calculate dbh from crown area rather than its reverse
    #
    # real(r8)            :: crown_area_to_dbh_exponent
    # real(r8)            :: spreadterm  ! Effective 2bh to crown area scaling factor
    # real(r8)            :: crown_reduction

    # ! default is to use the same exponent as the dbh to bleaf exponent so that per-plant
    # ! canopy depth remains invariant during growth, but allowed to vary via the
    # ! allom_blca_expnt_diff term (which has default value of zero)
    crown_area_to_dbh_exponent = d2bl_p2 + d2bl_ediff

    # ! ----------------------------------------------------------------------------------
    # ! The function c_area is called during the process of canopy position demotion
    # ! and promotion. As such, some cohorts are temporarily elevated to canopy positions
    # ! that are outside the number of alloted canopy spaces.  Ie, a two story canopy
    # ! may have a third-story plant, if only for a moment.  However, these plants
    # ! still need to generate a crown area to complete the promotion, demotion process.
    # ! So we allow layer index exceedence here and force it down to max.
    # ! (rgk/cdk 05/2017)
    # ! ----------------------------------------------------------------------------------

    # ! apply site-level spread elasticity to the cohort crown allometry term

    spreadterm = spread * d2ca_max + (1. - spread) * d2ca_min

    if not(inverse):
        c_area = spreadterm * dbh ** crown_area_to_dbh_exponent

        if(crowndamage > 1):
            crown_reduction =  GetCrownReduction(crowndamage)
            c_area = c_area * (1.0 - crown_reduction)

        return c_area

    else:
        if(crowndamage > 1):
            crown_reduction =  GetCrownReduction(crowndamage)
            c_area = c_area/(1.0 - crown_reduction)
        dbh = (c_area / spreadterm) ** (1./crown_area_to_dbh_exponent)
        return dbh
# ======================================================================================================================
def carea_3pwr(dbh,height,spread,dh2bl_p2,dh2bl_ediff, dh2ca_min,dh2ca_max,crowndamage,inverse = False):
    # !---~---
    # !    Calculate area of ground covered by entire cohort. (m2)
    # ! Function of DBH (cm), height (m), canopy spread (m/cm) and number of
    # ! individuals.
    # !---~---
    #
    # !--- List of arguments
    # real(r8)   , intent(inout) :: dbh         ! Diameter at breast/ref/ height     [   cm]
    # real(r8)   , intent(inout) :: height      ! Height                             [    m]
    # integer(i4), intent(in)    :: ipft        ! PFT index
    # real(r8)   , intent(in)    :: dbh_maxh    ! Minimum DBH at maximum height      [   cm]
    # real(r8)   , intent(in)    :: spread      ! site level relative spread score   [  0-1]
    # real(r8)   , intent(in)    :: dh2bl_p2    ! Exponent for size (bleaf)          [    -]
    # real(r8)   , intent(in)    :: dh2bl_ediff ! Difference in size exponent        [    -]
    #                                        !    between crown area and bleaf
    # real(r8)   , intent(in)    :: dh2ca_min   ! Minimum (closed forest) scaling    [    -]
    #                                        !    coefficient for crown area
    # real(r8)   , intent(in)    :: dh2ca_max   ! Maximum (savannah) scaling         [    -]
    #                                        !    coefficient for crown area
    # integer    , intent(in)    :: crowndamage ! Crown damage class                 [    -]
    #                                        !    [1: undamaged, >1: damaged]
    # real(r8)   , intent(inout) :: c_area      ! crown area for one plant           [   m2]
    # logical    , intent(in)    :: inverse     ! If true, calculate dbh from crown
    #                                        !    area rather than its reverse
    # !--- Local variables
    # real(r8) :: size            ! Size (Diameter^2 * Height)                       [cm2 m]
    # real(r8) :: dh2ca_p1        ! Effective scaling factor (crown area)            [    -]
    # real(r8) :: dh2ca_p2        ! Effective exponent (crown area)                  [    -]
    # real(r8) :: crown_reduction ! Crown area reduction due to damage.              [    -]
    # !---~---


    # !---~---
    # !   Define the scaling (log-intercept) and exponent (log-slope) parameters for
    # ! crown area. The scaling parameter accounts for the site-level spread elasticity.
    # ! The exponent is defined in terms of the leaf biomass exponent plus an offset
    # ! parameter (allom_blca_expnt_diff). This is done because the default in FATES is
    # ! for both exponents to be same (i.e., allom_blca_expnt_diff = 0.) so the per-plant
    # ! canopy area remains invariant during growth. However, allometric models in general
    # ! predict that leaf area grows faster than crown area.
    # !---~---
    dh2ca_p1 = spread * dh2ca_max + (1. - spread) * dh2ca_min
    dh2ca_p2 = dh2bl_p2 + dh2bl_ediff
    # !---~---

    # !---~---
    # !   Decide whether to use DBH and height to find crown area (default) or the
    # ! other way round.
    # !---~---
    if not(inverse):
        # !--- Find the maximum area
        size   = dbh * dbh * height
        c_area = dh2ca_p1 * size ** dh2ca_p2
        # !---~---
        #
        # !--- Reduce area if the crown is damaged.
        if (crowndamage > 1):
            crown_reduction =  GetCrownReduction(crowndamage)
            c_area = c_area * (1.0 - crown_reduction)
        # !---~---
    else:
        raise RuntimeError("Inverse case is not implemented for this function")

     # case (.true.)
     #    !--- Reduce area if the crown is damaged.
     #    if (crowndamage > 1) then
     #       call GetCrownReduction(crowndamage, crown_reduction)
     #       c_area = c_area * (1.0_r8 - crown_reduction)
     #    end if
     #    !---~---
     #
     #
     #    !---~---
     #    !   Find the size, then use a root-finding algorithm to find DBH.
     #    !---~---
     #    size = ( c_area / dh2ca_p1 ) ** ( 1.0_r8 / dh2ca_p2 )
     #    call size2dbh(size,ipft,dbh,dbh_maxh)
     #    !---~---
     # end select
     # !---~---

    return c_area
#=======================================================================================================================
def decay_coeff_vcmax(vcmax25top,slope_param,intercept_param):

    # ! ---------------------------------------------------------------------------------
    # ! This function estimates the decay coefficient used to estimate vertical
    # ! attenuation of properties in the canopy.
    # !
    # ! Decay coefficient (kn) is a function of vcmax25top for each pft.
    # !
    # ! Currently, this decay is applied to vcmax attenuation, SLA (optionally)
    # ! and leaf respiration (optionally w/ Atkin)
    # !
    # ! ---------------------------------------------------------------------------------
    #
    # !ARGUMENTS

    # real(r8),intent(in) :: vcmax25top
    # real(r8),intent(in) :: slope_param      ! multiplies vcmax25top
    # real(r8),intent(in) :: intercept_param  ! adds to vcmax25top
    #
    #
    # !LOCAL VARIABLES
    # ! -----------------------------------------------------------------------------------
    #
    # ! Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593 used
    # ! kn = 0.11. Here, we derive kn from vcmax25 as in Lloyd et al
    # ! (2010) Biogeosciences, 7, 1833-1859
    # ! This function is also used to vertically scale leaf maintenance
    # ! respiration.

    decay_coeff_vcmax = torch.exp(slope_param * vcmax25top - intercept_param)

    return decay_coeff_vcmax
#=======================================================================================================================
def carea_allom(dbh, h , prt_params,site_spread,crowndamage,nplant = 1):

     # real(r8),intent(inout) :: dbh          ! plant diameter at breast (reference) height [cm]
     # real(r8),intent(in)    :: site_spread  ! site level spread factor (crowdedness)
     # real(r8),intent(in)    :: nplant       ! number of plants [1/ha]
     # integer(i4),intent(in) :: ipft         ! PFT index
     # integer(i4),intent(in) :: crowndamage  ! crown damage class [1: undamaged, >1: damaged]
     # real(r8),intent(inout) :: c_area       ! crown area per cohort (m2)
     # logical,optional,intent(in) :: inverse ! if true, calculate dbh from crown area
     #                                        ! instead of crown area from dbh
     #
     # real(r8)               :: dbh_eff      ! Effective diameter (cm)
     # real(r8)               :: height       ! height
     # logical                :: do_inverse   ! local copy of the inverse argument
     #                                        ! defaults to false
     # logical                :: capped_allom ! if we are using an allometry that caps
     #                                        ! crown area at height, we need to make
     #                                        ! special considerations

    dbh_maxh    = prt_params['dbh_maxh']
    allom_lmode = prt_params['allom_lmode']
    d2bl_p2     = prt_params['d2bl2']# prt_params['d2bl_p2']
    d2bl_ediff  = prt_params['d2bl_ediff']
    d2ca_min    = prt_params['d2ca_min']
    d2ca_max    = prt_params['d2ca_max']

    carea = torch.full_like(dbh, fates_unset_r8)
    for mode in range(1, 5):
        mask = (allom_lmode == mode).bool()#.astype(bool)#
        if mode ==1:
            dbh_eff = torch.minimum(dbh,dbh_maxh)
            carea__ = carea_2pwr(dbh_eff,site_spread,d2bl_p2,d2bl_ediff,d2ca_min,d2ca_max, crowndamage)
            capped_allom = True
        elif mode ==2:
            carea__ = carea_2pwr(dbh,site_spread,d2bl_p2,d2bl_ediff,d2ca_min,d2ca_max, crowndamage)
            capped_allom = False
        elif mode ==3:
            dbh_eff = torch.minimum(dbh,dbh_maxh)
            carea__ = carea_2pwr(dbh_eff,site_spread,d2bl_p2,d2bl_ediff,d2ca_min,d2ca_max, crowndamage)
            capped_allom = True
        elif mode ==4:
            dbh_eff = torch.minimum(dbh,dbh_maxh)
            # call h_allom(dbh,ipft,height)
            carea__ =  carea_3pwr(dbh_eff,h, site_spread,d2bl_p2, d2bl_ediff, d2ca_min,d2ca_max,crowndamage)
            capped_allom = True
        else:
            raise RuntimeError("This allometry mode is not defined yet")

        carea = torch.where(mask, carea__, carea)

       # if (capped_allom .and. do_inverse) then
       #    if (dbh_eff .lt. dbh_maxh) then
       #       dbh = dbh_eff
       #    else
       #       ! In this situation, we are signaling to the
       #       ! calling routine that we we cannot calculate
       #       ! dbh from crown area, because we have already
       #       ! hit the area cap, and the two are not proportional
       #       ! anymore.  hopefully, the calling routine has an alternative
       #       dbh = fates_unset_r8
       #    endif
       # endif

    carea = carea * nplant

    return carea
#===================================================================================================
def UpdateAllom(pl_dbh, pl_height, pl_sapwood, pl_leafarea, prt_params):

    h, dhdd = h_allom(d = pl_dbh,prt_params = prt_params)


    bl, dbldd = bleaf(d=pl_dbh,
                      prt_params=prt_params,
                      crowndamage=1,
                      canopy_trim=max(1.0, min_trim),
                      elongf_leaf=1)

    # get the calculated plant height
    #================================
    h_mask = torch.isnan(pl_height)#(pl_height == -1).bool()
    h      = torch.where(h_mask, h, pl_height)

    # get the calculated sapwood area
    #================================
    sapw_area,_,_ = bsap_ltarg_slatop(prt_params,h,dhdd,bl,dbldd)
    sapw_area_mask= torch.isnan(pl_sapwood)
    sapw_area     = torch.where(sapw_area_mask, sapw_area, pl_sapwood)

    # get the calculated leaf area
    # ================================
    leafarea     = bl * 1000 * prt_params['slatop'] # 1000 to convert from kgC to gC * sla (m2/gC)
    leafarea_mask= torch.isnan(pl_leafarea)
    leafarea     = torch.where(leafarea_mask, leafarea, pl_leafarea)

    # get the crown area
    # ==================
    c_area = carea_allom(dbh        =pl_dbh,
                         h          =h,
                         prt_params =prt_params,
                         site_spread=0.5,
                         crowndamage=1,
                         nplant     =1)
    # get the leaf area index
    # =======================
    lai = leafarea / c_area


    return h, sapw_area, leafarea, c_area, lai