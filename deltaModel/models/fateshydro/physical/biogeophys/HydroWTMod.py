
import torch
from conf.Constants import fates_r8 as r8, fates_unset_r8, nearzero


# This module contains all unit functions associated with Water Transfer (WTFs).
# These functions may be applied to xylems, stems, etc, not limited to soils.

# Define some constant parameters
min_ftc: r8         = 0.00001  # Minimum allowed fraction of total conductance

# Bounds on saturated fraction for linear interpolation
min_sf_interp: r8   = 0.01
max_sf_interp: r8   = 0.998

# Smoothing factors for the capillary-elastic and elastic-caviation regions
quad_a1: r8         = 0.80  # Smoothing factor "A" term in the capillary-elastic region
quad_a2: r8         = 0.99  # Smoothing factor "A" term in the elastic-caviation region


class wrf_type:
    def __init__(self):

        # self.psi_max     = None     # psi matching max_sf_interp where we start linear interp
        # self.psi_min     = None     # psi matching min_sf_interp
        # self.dpsidth_max = None     # dpsi_dth where we start linear interp
        # self.dpsidth_min = None     # dpsi_dth where we start min interp
        # self.th_min      = None     # vwc matching min_sf_interp where we start linear interp
        # self.th_max      = None     # vwc matching max_sf_interp where we start linear interp
        return

    def th_from_psi(self, psi):
        raise NotImplementedError("th_from_psi method should be overridden in derived classes")

    def psi_from_th(self, th):
        raise NotImplementedError("psi_from_th method should be overridden in derived classes")

    def dpsidth_from_th(self, th):
        raise NotImplementedError("dpsidth_from_th method should be overridden in derived classes")

    def set_wrf_param(self, params_in):
        raise NotImplementedError("set_wrf_param method should be overridden in derived classes")

    def get_thsat(self):
        raise NotImplementedError("get_thsat method should be overridden in derived classes")


    def psi_linear_sat(self, th):
        """
        Calculate psi (matric potential) in the linear range above saturation.
        Parameters:
            th (float): Volumetric water content [m³/m³]
        Returns:
            float: Matric potential [MPa]
        """
        psi = self.psi_max + self.dpsidth_max * (th - self.th_max)
        return psi

    def psi_linear_res(self, th):
        """
        Calculate psi (matric potential) in the linear range below residual.
        Parameters:
            th (float): Volumetric water content [m³/m³]
        Returns:
            float: Matric potential [MPa]
        """
        psi = self.psi_min + self.dpsidth_min * (th - self.th_min)
        return psi

    def th_linear_sat(self, psi):
        """
        Calculate theta (volumetric water content) from psi in the linear range above saturation.
        Parameters:
            psi (float): Matric potential [MPa]
        Returns:
            float: Volumetric water content [m³/m³]
        """
        th = self.th_max + (psi - self.psi_max) / self.dpsidth_max
        return th

    def th_linear_res(self, psi):
        """
        Calculate theta (volumetric water content) from psi in the linear range below saturation.
        Parameters:
            psi (float): Matric potential [MPa]
        Returns:
            float: Volumetric water content [m³/m³]
        """
        th = self.th_min + (psi - self.psi_min) / self.dpsidth_min
        return th

    def set_min_max(self, th_res, th_sat):
        # This routine uses max_sf_interp and min_sf_interp
        # to define the bounds of where the linear ranges start and stop

        self.th_max         = max_sf_interp * (th_sat - th_res) + th_res
        self.th_min         = min_sf_interp * (th_sat - th_res) + th_res

        # Calculate psi_max and dpsidth_max
        #if torch.is_tensor(self.th_max):
        self.psi_max        = self.psi_from_th(self.th_max - torch.finfo(self.th_max.dtype).tiny)#tiny(self.th_max)
        self.dpsidth_max    = self.dpsidth_from_th(self.th_max - torch.finfo(self.th_max.dtype).tiny)
        #else:
            # self.psi_max        = self.psi_from_th(self.th_max - torch.finfo(type(self.th_max)).tiny)#tiny(self.th_max)
            # self.dpsidth_max    = self.dpsidth_from_th(self.th_max - torch.finfo(type(self.th_max)).tiny)

        # Calculate psi_min and dpsidth_min
        # if torch.is_tensor(self.th_min):
        self.psi_min        = self.psi_from_th(self.th_min + torch.finfo(self.th_min.dtype).tiny)
        self.dpsidth_min    = self.dpsidth_from_th(self.th_min + torch.finfo(self.th_min.dtype).tiny)
        # else:
            # self.psi_min        = self.psi_from_th(self.th_min + torch.finfo(type(self.th_min)).tiny)
            # self.dpsidth_min    = self.dpsidth_from_th(self.th_min + torch.finfo(type(self.th_min)).tiny)

    def get_thmin(self):
        """
        Get the minimum value of theta (volumetric water content).
        """
        return self.th_min
########################################################################################################################
class wrf_type_vg(wrf_type):

    def __init__(self):
        super().__init__()  # Call the parent class constructor
        # self.alpha = None       #! Inverse air entry parameter         [m3/Mpa]
        # self.n_vg = None        #! pore size distribution parameter, psd in original code
        # self.m_vg = None        #! m in van Genuchten 1980, also a pore size distribtion parameter , 1-m in original code
        # self.th_sat = None      #! Saturation volumetric water content [m3/m3]
        # self.th_res = None      #! Residual volumetric water content   [m3/m3]
        return

    def th_from_psi(self, psi):
        #     ! Van Genuchten (1980) calculation of volumetric water content (theta)
        #     ! from matric potential.
        m = self.m_vg
        n = self.n_vg

        satfrac = (1.0 + (-self.alpha * psi) ** n) ** (-m)
        mask1   = psi > self.psi_max; mask1 = mask1.type(torch.uint8)
        mask2   = psi < self.psi_min; mask2 = mask2.type(torch.uint8)
        th      = satfrac * (self.th_sat - self.th_res) + self.th_res
        th      = torch.where(mask1,
                              self.th_linear_sat(psi),
                              torch.where(mask2, self.th_linear_res(psi), th))

        return th


    def psi_from_th(self, th):
        # !------------------------------------------------------------------------------------
        # ! saturation fraction is the origial equation in vg 1980, we just
        # ! need to invert it:
        # ! (note "psd" is the pore-size-distribution parameter, equivalent to "n" from the
        # ! manuscript.)
        # !
        # ! satfrac = (1._r8 + (alpha*psi)**psd)**(-m)
        # !
        # ! *also modified to accomodate linear pressure regime for super-saturation
        # ! -----------------------------------------------------------------------------------

        m       = self.m_vg
        n       = self.n_vg
        satfrac = (th - self.th_res) / (self.th_sat - self.th_res)
        mask1   = th > self.th_max
        mask2   = th < self.th_min

        psi     = -(1.0 / self.alpha) * (satfrac ** (1.0 / (-m)) - 1.0) ** (1.0 / n)
        if hasattr(self, 'psi_max') and hasattr(self, 'psi_min'):
            psi = torch.where(mask1, self.psi_linear_sat(th), torch.where(mask2, self.psi_linear_res(th), psi))
        else:
            psi = psi

        return psi

    def dpsidth_from_th(self, th):
        a1 = 1.0 / self.alpha
        m1 = 1.0 / self.n_vg
        m2 = -1.0 / self.m_vg
        satfrac      = (th - self.th_res) / (self.th_sat - self.th_res)
        dsatfrac_dth = 1.0 / (self.th_sat - self.th_res)
        mask1        = th > self.th_max
        mask2        = th < self.th_min

        dpsidth      = -m1 * a1 * (satfrac ** m2 - 1.0) ** (m1 - 1.0) * m2 * satfrac ** (m2 - 1.0) * dsatfrac_dth
        if hasattr(self, 'dpsidth_max') and hasattr(self, 'dpsidth_min'):
            dpsidth      = torch.where(mask1,self.dpsidth_max, torch.where(mask2,self.dpsidth_min, dpsidth))
        else:
            dpsidth = dpsidth

        return dpsidth

    def set_wrf_param(self, params_in):
        # if len(params_in) != 5:
        #     raise ValueError("params_in should have exactly 5 elements")

        self.alpha  = params_in[0]
        self.n_vg   = params_in[1]
        self.m_vg   = params_in[2]
        self.th_sat = params_in[3]
        self.th_res = params_in[4]

        self.set_min_max(self.th_res, self.th_sat)

    def get_thsat(self):

        return self.th_sat
########################################################################################################################
class wrf_type_cch(wrf_type):

    def __init__(self):
        super().__init__()  # Call the parent class constructor
        #real(r8):: th_sat   ! Saturation volumetric water content[m3 / m3]
        #real(r8):: psi_sat  ! Bubbling pressure(potential at saturation) [Mpa]
        #real(r8):: beta     ! Clapp - Hornberger "beta" parameter[-]
        return

    def th_from_psi(self, psi):

        th    = self.th_sat * (psi / self.psi_sat) ** (-1.0 / self.beta)
        mask  = psi > self.psi_max
        if hasattr(self, 'psi_max'):
            th    = torch.where(mask,self.th_max + (psi - self.psi_max) / self.dpsidth_max, th)
        else:
            th = th

        return th

    def psi_from_th(self,th):

        psi  = self.psi_sat * (th / self.th_sat) ** (-self.beta)
        mask = th > self.th_max
        if hasattr(self, 'psi_max'):
            psi  = torch.where(mask, self.psi_max + self.dpsidth_max * (th - max_sf_interp * self.th_sat), psi)
        else:
            psi = psi

        return psi

    def dpsidth_from_th(self,th):

        dpsidth = -self.beta * self.psi_sat / self.th_sat * (th / self.th_sat) ** (-self.beta - 1.0)
        mask    = th > self.th_max
        if hasattr(self, 'dpsidth_max'):
            dpsidth = torch.where(mask,self.dpsidth_max, dpsidth)
        else:
            dpsidth = dpsidth

        return dpsidth

    def set_wrf_param(self, params_in):
        self.th_sat   = params_in[0]
        self.psi_sat  = params_in[1]
        self.beta     = params_in[2]
        
        # Set DERIVED constants used for interpolating in extreme ranges
        self.th_max      = max_sf_interp * self.th_sat
        self.psi_max     = self.psi_from_th(self.th_max - torch.finfo(self.th_max.dtype).tiny)
        self.dpsidth_max = self.dpsidth_from_th(self.th_max - torch.finfo(self.th_max.dtype).tiny)
        self.th_min      = fates_unset_r8
        self.psi_min     = fates_unset_r8
        self.dpsidth_min = fates_unset_r8

        self.attrs = {"th_sat": self.th_sat, "psi_sat": self.psi_sat, "beta": self.beta,
                      "th_max": self.th_max, "psi_max": self.psi_max, "dpsidth_max" :self.dpsidth_max}
        return
    def get_thsat(self):
        return self.th_sat

    def subsetAttrs(self, indices):
        return {attr: getattr(self, attr)[indices] for attr in ['th_sat', 'psi_sat', 'beta', 'th_max', 'psi_max', 'dpsidth_max'] if hasattr(self, attr)}

    def updateAttrs(self, subsetParams):
        for attr, value in subsetParams.items():
            setattr(self, attr, value)
            self.attrs[attr] = value
        return
########################################################################################################################
class wrf_type_smooth_cch(wrf_type):
    # ! =====================================================================================
    # ! Type1 Smooth approximation of Clapp-Hornberger and Campbell (CCH) water retention and conductivity functions
    # ! Bisht et al. Geosci. Model Dev., 11, 4085–4102, 2018
    # ! =====================================================================================

    def __init__(self):
        super().__init__()  # Call the parent class constructor
        # real(r8) :: th_sat   ! Saturation volumetric water content         [m3/m3]
        # real(r8) :: psi_sat  ! Bubbling pressure (potential at saturation) [Mpa]
        # real(r8) :: beta     ! Clapp-Hornberger "beta" parameter           [-]
        # real(r8) :: scch_pu  ! An estimated breakpoint capillary pressure, below which the specified water retention curve is applied. It is also the lower limit when the smoothing function is applied. [Mpa]
        # real(r8) :: scch_ps  ! An estimated breakpoint capillary pressure, an upper limit where smoothing funciton is applied. [Mpa]
        # real(r8) :: scch_b2  ! constant coefficient of the quadratic term in the smoothing polynomial function [-]
        # real(r8) :: scch_b3  ! constant coefficient of the cubic term in the smoothing polynomial function [-]
        return

    def th_from_psi(self, psi):

        alpha   = -1.0 / self.psi_sat
        lamda   = 1.0  / self.beta #change from lambda to lamda to avoid a reserved word
        pc      = psi
        deltaPc = pc - self.scch_ps
        mask1   = pc <= self.scch_pu
        mask2   = pc < self.scch_ps

        sat     = torch.zeros_like(pc) + 1.0
        sat     = torch.where(mask1,  (-alpha * pc) ** (- lamda), torch.where(mask2,1.0 + deltaPc * deltaPc * (self.scch_b2 + deltaPc * self.scch_b3), sat))

        th = sat * self.th_sat

        return th

    def psi_from_th(self, th):
        relTol  = 1e-9
        sat_res = 0.0
        alpha   = -1.0 / self.psi_sat
        lamda   = 1.0 / self.beta

        sat     = torch.clamp(th / self.th_sat, min = 1e-6)

        if sat < 1.0:
            Se = sat
            pc = -(Se ** (-1.0 / lamda)) / alpha

            if pc > self.scch_pu:
                if self.scch_b2 == 0.0:
                    pc = self.scch_ps - ((1.0 - Se) / self.scch_b3) ** (1.0 / 3.0)
                elif self.scch_b3 == 0.0:
                    pc = self.scch_ps - torch.sqrt((Se - 1.0) / self.scch_b2)
                else:
                    xL   = self.scch_pu - self.scch_ps
                    xR   = 0.0
                    xc   = pc - self.scch_ps
                    iter = 0
                    dx   = 1e20  # Something large



                    while abs(dx) >= -relTol * self.scch_pu:
                        iter += 1

                        if xc <= xL or xc >= xR:
                            xc = xL + 0.5 * (xR - xL)

                        dx    = self.scch_b3 * xc
                        resid = xc * xc * (self.scch_b2 + dx) + 1.0 - Se
                        dx    = resid / (xc * (2.0 * self.scch_b2 + 3.0 * dx))

                        if resid > 0.0:
                            xR = xc
                        else:
                            xL = xc

                        xc = xc - dx

                        if iter > 10000:
                            raise RuntimeError("psi_from_th_smooth_cch iteration not converging")

                    pc = xc + self.scch_ps
        else:
            pc = 0.0

        psi = pc
        return psi

    def dpsidth_from_th(self, th):


        sat_res = 0.0
        alpha     = -1.0 / self.psi_sat
        lambda_   = 1.0 / self.beta
        pc        = 1.0 * self.psi_from_th(th)
        deltaPc   = pc - self.scch_ps

        mask1   = pc <= self.scch_pu
        mask2   = pc < self.scch_ps

        Se      = torch.where(mask1,(-alpha * pc) ** (- lambda_) , 1.0 + deltaPc * deltaPc * (self.scch_b2 + deltaPc * self.scch_b3))
        sat = sat_res + (1.0 - sat_res) * Se

        dSe_dpc = torch.where(mask1,  - lambda_ * Se / pc,deltaPc * (2 * self.scch_b2 + 3 * deltaPc * self.scch_b3))
        dsat_dp = (1.0 - sat_res) * dSe_dpc
        dpsidth = self.dpsidth_max
        dpsidth = torch.where(mask1, 1.0 / (dsat_dp * self.th_sat), torch.where(mask2, 1.0 / (dsat_dp * self.th_sat), dpsidth))


        return dpsidth

    def set_wrf_param(self, params_in):
        #  integer  :: styp    ! an option to force constant coefficient of the quadratic
        #                      ! term 0 (styp = 1) or to force the constant coefficient of
        #                      ! the cubic term 0 (styp/=2)

        #  real(r8) :: th_max  ! saturated water content [-]

        # ! !LOCAL VARIABLES:
        #  real(r8) :: pu                ! an estimated breakpoint at which the constant
        #                                ! coefficient of the quadratic term (styp=2)
        #                                ! or the cubic term (styp/=2) is 0 [Mpa]
        # real(r8):: bcAtPu              ! working local
        # real(r8):: lambdaDeltaPuOnPu   ! working local
        # real(r8):: oneOnDeltaPu        ! working local
        # real(r8):: lambda              ! working local, inverse of Clapp and Hornberger "b"
        # real(r8):: alpha               ! working local
        # real(r8):: ps                  ! working local, 90 % of entry pressure[Mpa]


        self.th_sat       = params_in[0]
        self.psi_sat      = params_in[1]
        self.beta         = params_in[2]
        styp              = int(params_in[3])

        alpha        = -1.0 / self.psi_sat
        lamda        = 1.0 / self.beta
        ps           = -0.9 / alpha
        self.scch_ps = ps

        if styp == 1:

            pu = findGu_SBC_zeroCoeff(lamda , torch.tensor([3]) , -alpha * ps) / (-alpha)
            self.scch_pu = pu

            bcAtPu           = (-alpha * pu) ** (- lamda )
            lamdaDeltaPuOnPu = lamda * (1.0 - ps / pu)
            oneOnDeltaPu     = 1.0 / (pu - ps)

            self.scch_b2 = 0.0
            self.scch_b3 = (2.0 - bcAtPu * (2.0 + lamdaDeltaPuOnPu)) * oneOnDeltaPu ** 3
            if torch.any(self.scch_b3 <= 0.0):
                raise ValueError('set_wrf_param_smooth_cch b3 <=0', pu, ps, alpha, lamda , oneOnDeltaPu, lamdaDeltaPuOnPu, bcAtPu, self.psi_sat)

        else:
            pu           = findGu_SBC_zeroCoeff(lamda , torch.tensor([2]), -alpha * ps) / (-alpha)
            self.scch_pu = pu

            bcAtPu           = (-alpha * pu) ** (- lamda )
            lamdaDeltaPuOnPu = lamda * (1.0 - ps / pu)
            oneOnDeltaPu     = 1.0 / (pu - ps)
            self.scch_b2     = -(3.0 - bcAtPu * (3.0 + lamdaDeltaPuOnPu)) * oneOnDeltaPu ** 2

            if torch.any(self.scch_b2 >= 0.0):
                raise ValueError('set_wrf_param_smooth_cch b2 <= 0')

            self.scch_b3 = 0.0

        self.th_max      = max_sf_interp * self.th_sat
        self.psi_max     = self.psi_from_th(self.th_max - torch.finfo(type(self.th_max)).tiny)
        self.dpsidth_max = self.dpsidth_from_th(self.th_max - torch.finfo(type(self.th_max)).tiny)
        self.th_min      = 1.0e-8
        self.psi_min     = fates_unset_r8
        self.dpsidth_min = fates_unset_r8
    
        return


    def get_thsat(self):
            return self.th_sat

########################################################################################################################
class wrf_type_tfs(wrf_type):

    # real(r8) :: th_sat   ! Saturation volumetric water content         [m3/m3]
    # real(r8) :: th_res   ! Residual volumentric water content          [m3/m3]
    # real(r8) :: pinot    ! osmotic potential at full turger            [MPa]
    # real(r8) :: epsil    ! bulk elastic modulus                        [MPa]
    # real(r8) :: rwc_ft   ! RWC @ full turgor, (elastic drainage begins)[-]
    # real(r8) :: cap_corr ! correction for nonzero psi0x
    # real(r8) :: cap_int  ! intercept of capillary region of curve
    # real(r8) :: cap_slp  ! slope of capillary region of curve
    # integer  :: pmedia   ! self describing porous media index

    def __init__(self):
        super().__init__()  # Call the parent class constructor

        # real(r8) :: th_sat   ! Saturation volumetric water content         [m3/m3]
        # real(r8) :: th_res   ! Residual volumentric water content          [m3/m3]
        # real(r8) :: pinot    ! osmotic potential at full turger            [MPa]
        # real(r8) :: epsil    ! bulk elastic modulus                        [MPa]
        # real(r8) :: rwc_ft   ! RWC @ full turgor, (elastic drainage begins)[-]
        # real(r8) :: cap_corr ! correction for nonzero psi0x
        # real(r8) :: cap_int  ! intercept of capillary region of curve
        # real(r8) :: cap_slp  ! slope of capillary region of curve
        # integer  :: pmedia   ! self describing porous media index

        return

    def th_from_psi(self, psi):
        # ! !LOCAL VARIABLES:
        # real(r8) :: lower                ! lower bound of initial estimate         [m3 m-3]
        # real(r8) :: upper                ! upper bound of initial estimate         [m3 m-3]

        # real(r8) :: satfrac              ! soil saturation fraction                [0-1]
        # real(r8) :: psi_check
        # Initialize th tensor
        th = torch.zeros_like(psi)

        # Masks for different conditions
        mask_psi_max = psi > self.psi_max; mask_psi_max  = mask_psi_max.bool()
        mask_psi_min = psi < self.psi_min; mask_psi_min  = mask_psi_min.bool()
        mask_else    = ~(mask_psi_max | mask_psi_min); mask_else  = mask_else.bool()

        # Handling psi > psi_max or < psi_min
        th = torch.where(mask_psi_max,  self.th_linear_sat(psi),
                         torch.where(mask_psi_min,self.th_linear_res(psi), th))

        # Handling else condition
        if mask_else.any():
            lower = self.th_min - 1e-9
            upper = self.th_max + 1e-9

            # Assumed bisect_pv method is adapted for tensors
            th = torch.where(mask_else,  self.bisect_pv(lower, upper, psi, mask = mask_else), th)
            #th[mask_else] = self.bisect_pv(lower, upper, psi[mask_else])

            # Check psi values
            psi_check = self.psi_from_th(th)[mask_else]
            if (psi_check > -1e-8).any():
                raise ValueError("bisect_pv returned positive value for water potential")

        return th

    def psi_from_th(self, th):
        satfrac = (th - self.th_res) / (self.th_sat - self.th_res)

        # Initialize psi tensor
        psi = torch.zeros_like(th)

        # Masks for different conditions
        mask_th_max = th > self.th_max
        mask_th_min = th < self.th_min
        mask_else = ~(mask_th_max | mask_th_min)

        if hasattr(self, 'psi_max') and hasattr(self, 'psi_min'):
            psi = torch.where(mask_th_max, self.psi_linear_sat(th), torch.where(mask_th_min,self.psi_linear_res(th), psi))

        # Handling else condition
        if mask_else.any():
            th_corr = th * self.cap_corr

            # Assumed solutepsi, pressurepsi, and capillarypsi methods are adapted for tensors
            psi_sol   = solutepsi(th_corr, self.rwc_ft, self.th_sat, self.th_res, self.pinot)
            psi_press = pressurepsi(th_corr, self.rwc_ft, self.th_sat, self.th_res, self.pinot, self.epsil)

            psi_elastic = psi_sol + psi_press

            psi_capelast  = torch.zeros_like(psi_elastic)
            psi_capillary = torch.zeros_like(psi_elastic)
            b = torch.zeros_like(psi_elastic)
            c = torch.zeros_like(psi_elastic)

            # Handling porous media types
            mask_pmedia_1 = self.pmedia == 1
            mask_pmedia_4 = (self.pmedia <= 4) & (~mask_pmedia_1)
            mask_pmedia_else = ~(mask_pmedia_1 | mask_pmedia_4)

            # pmedia == 1
            psi_capelast = torch.where(mask_pmedia_1, psi_elastic ,psi_capelast)

            # pmedia <= 4
            if mask_pmedia_4.any():

                psi_capillary = torch.where(mask_pmedia_4, capillarypsi(th_corr, self.th_sat, self.cap_int,
                                                            self.cap_slp), psi_capillary)
                b            = torch.where(mask_pmedia_4, -1.0 * (psi_capillary + psi_elastic), b)
                c            = torch.where(mask_pmedia_4, psi_capillary * psi_elastic, c)
                psi_capelast = torch.where(mask_pmedia_4, (-b - torch.sqrt(b** 2 - 4.0 * quad_a1 * c)) / (2.0 * quad_a1),
                                           psi_capelast)

            # Handling invalid pmedia
            if mask_pmedia_else.any():
                raise ValueError("TFS WRF was called for an ineligible porous media")

            # Smoothing with cavitation
            psi_cavitation = psi_sol
            b              = -1.0 * (psi_capelast + psi_cavitation)
            c              = psi_capelast * psi_cavitation
            mask_else      = mask_else.bool()

            #psi[mask_else] = (-b + torch.sqrt(b ** 2 - 4.0 * quad_a2 * c)) / (2.0 * quad_a2)
            psi       = torch.where(mask_else,  (-b + torch.sqrt(b ** 2 - 4.0 * quad_a2 * c)) / (2.0 * quad_a2), psi)


        return psi

    def dpsidth_from_th(self, th):

        # Initialize dpsidth tensor
        dpsidth = torch.zeros_like(th)

        # Masks for different conditions
        mask_th_max = th > self.th_max
        mask_th_min = th < self.th_min
        mask_else = ~(mask_th_max | mask_th_min)

        # Handling th > th_max
        if hasattr(self, 'dpsidth_max') and hasattr(self, 'dpsidth_min'):
            dpsidth = torch.where(mask_th_max, self.dpsidth_max, torch.where(mask_th_min, self.dpsidth_min, dpsidth))


        # Handling else condition
        if mask_else.any():
            th_corr = th * self.cap_corr

            # Assumed methods are adapted for tensors
            psi_sol   = solutepsi(th_corr, self.rwc_ft, self.th_sat, self.th_res, self.pinot)
            psi_press = pressurepsi(th_corr, self.rwc_ft, self.th_sat, self.th_res, self.pinot, self.epsil)
            dsol_dth  = dsolutepsidth(th, self.th_sat, self.th_res, self.rwc_ft, self.pinot)
            dpress_dth= dpressurepsidth(self.th_sat, self.th_res, self.rwc_ft, self.epsil)

            delast_dth = dsol_dth + dpress_dth
            psi_elastic = psi_sol + psi_press

            psi_capelast = torch.zeros_like(psi_elastic)
            dcapelast_dth = torch.zeros_like(psi_elastic)

            # Handling porous media types
            mask_pmedia_1 = self.pmedia == 1
            mask_pmedia_4 = (self.pmedia <= 4) & (~mask_pmedia_1)
            mask_pmedia_else = ~(mask_pmedia_1 | mask_pmedia_4)

            # pmedia == 1
            psi_capelast = torch.where(mask_pmedia_1, psi_elastic, psi_capelast)
            dcapelast_dth= torch.where(mask_pmedia_1, delast_dth, dcapelast_dth)

            # pmedia <= 4
            if mask_pmedia_4.any():
                psi_capillary = capillarypsi(th, self.th_sat, self.cap_int, self.cap_slp)
                b             = -1.0 * (psi_capillary + psi_elastic)
                c             = psi_capillary * psi_elastic
                psi_capelast = torch.where(mask_pmedia_4, (-b - torch.sqrt(b ** 2 - 4.0 * quad_a1 * c)) / (2.0 * quad_a1),
                                           psi_capelast)


                dcap_dth = dcapillarypsidth(self.cap_slp, self.th_sat)
                dbdth    = -1.0 * (delast_dth + dcap_dth)
                dcdth    = psi_elastic * dcap_dth + delast_dth * psi_capillary

                dcapelast_dth = torch.where(mask_pmedia_4,  (1.0 / (2.0 * quad_a1)) * (
                    -dbdth - 0.5 * ((b ** 2 - 4.0 * quad_a1 * c) ** -0.5) *
                    (2.0 * b * dbdth - 4.0 * quad_a1 * dcdth)), dcapelast_dth)


            # Handling invalid pmedia
            if mask_pmedia_else.any():
                raise ValueError("TFS WRF was called for an ineligible porous media")

            # Smoothing with cavitation
            psi_cavitation = psi_sol
            b               = -1.0 * (psi_capelast + psi_cavitation)
            c               = psi_capelast * psi_cavitation
            dcav_dth        = dsol_dth
            dbdth           = -1.0 * (dcapelast_dth + dcav_dth)
            dcdth           = psi_capelast * dcav_dth + dcapelast_dth * psi_cavitation

            mask_else = mask_else.bool()
            dpsidth   = torch.where(mask_else,(1.0 / (2.0 * quad_a2)) * (
                -dbdth + 0.5 * ((b ** 2 - 4.0 * quad_a2 * c) ** -0.5) *
                (2.0 * b * dbdth - 4.0 * quad_a2 * dcdth)), dpsidth )


        return dpsidth

    def set_wrf_param(self, params_in):

        self.th_sat = params_in[0]
        self.th_res = params_in[1]
        self.pinot  = params_in[2]
        self.epsil  = params_in[3]
        self.rwc_ft = params_in[4]
        self.cap_corr= params_in[5]
        self.cap_int= params_in[6]
        self.cap_slp= params_in[7]
        self.pmedia = params_in[8]

        self.set_min_max(self.th_res, self.th_sat)
        self.attrs = {'th_sat': self.th_sat, 'th_res': self.th_res    , 'pinot': self.pinot    , 'epsil': self.epsil,
                      'rwc_ft': self.rwc_ft, 'cap_corr': self.cap_corr, 'cap_int': self.cap_int, 'cap_slp': self.cap_slp,
                      'pmedia': self.pmedia, 'th_max' : self.th_max   , 'th_min': self.th_min  , 'psi_max': self.psi_max,
                      'dpsidth_max': self.dpsidth_max, 'psi_min': self.psi_min, 'dpsidth_min': self.dpsidth_min}
        return

    def get_thsat(self):
        return self.th_sat

    def bisect_pv(self, lower, upper, psi, max_iter = 1000, xtol = 1.e-16,ytol = 1.e-8, mask= None):

        # ! !DESCRIPTION: Bisection routine for getting the inverse of the plant PV curve.
        # !  An analytical solution is not possible because quadratic smoothing functions
        # !  are used to remove discontinuities in the PV curve.

        # real(r8)      , intent(inout)  :: lower       ! lower bound of estimate           [m3 m-3]
        # real(r8)      , intent(inout)  :: upper       ! upper bound of estimate           [m3 m-3]
        # real(r8)      , intent(in)     :: psi         ! water potential                   [MPa]
        # real(r8)      , intent(in)     :: psi         ! water potential                   [MPa]

        # ! !LOCAL VARIABLES:
        #  real(r8) :: x_new                  ! new estimate for x in bisection routine
        #  real(r8) :: y_lo                   ! corresponding y value at lower
        #  real(r8) :: f_lo                   ! y difference between lower bound guess and target y
        #  real(r8) :: y_hi                   ! corresponding y value at upper
        #  real(r8) :: f_hi                   ! y difference between upper bound guess and target y
        #  real(r8) :: y_new                  ! corresponding y value at x.new
        #  real(r8) :: f_new                  ! y difference between new y guess at x.new and target y
        #  real(r8) :: chg                    ! difference between x upper and lower bounds (approach 0 in bisection)
        #  integer  :: nitr                   ! number of iterations


        #xtol = 1.e-16                         # ! error tolerance for th[m3 m - 3]
        #ytol = 1.e-8                          # ! error tolerance for psi         [MPa]

        """
              Bisection routine for getting the inverse of the plant PV curve.
        """
        y_lo = self.psi_from_th(lower)
        y_hi = self.psi_from_th(upper)

        f_lo = y_lo - psi
        f_hi = y_hi - psi
        chg  = upper - lower
        mid  = 0.5 * (lower + upper)

        if (torch.abs(chg) <= xtol).all():
            print("Cannot enter solver iterations since upper bound = lower bound")
            x_new = mid
        else:

            x_new = torch.zeros_like(psi)
            for nitr in range(max_iter):
                mid = 0.5 * (lower + upper)
                y_mid = self.psi_from_th(mid)
                f_mid = y_mid - psi


                # Update bounds and function values
                lower_update = (f_lo * f_mid) < 0
                upper_update = (f_hi * f_mid) < 0

                lower[upper_update] = mid[upper_update]
                upper[lower_update] = mid[lower_update]
                chg = upper - lower

                # Check for convergence
                converged = (torch.abs(f_mid) <= ytol) | (torch.abs(chg) <= xtol)
                x_new[converged] = mid[converged]
                # This condition was added to solve only wherever mask is TRUE (Avoid long un-necessary iterations)
                if mask is not None:
                    if converged[mask].all():
                        break
                else:
                    if converged.all():
                        break

                if (nitr == max_iter - 1):
                    print('Warning: number of iteraction exceeds maximum iterations')
                    print("f(x) = ", torch.abs(f_mid).max())
                    x_new = mid

        th = x_new

        return th

    def subsetAttrs(self, indices):
        return {attr: getattr(self, attr)[indices] for attr in
                ['th_sat', 'th_res', 'pinot', 'epsil', 'rwc_ft', 'cap_corr', 'cap_int', 'cap_slp', 'pmedia',
                 'th_max', 'th_min', 'psi_max', 'dpsidth_max', 'psi_min', 'dpsidth_min'] if hasattr(self, attr)}
    def updateAttrs(self, subsetParams):
        for attr, value in subsetParams.items():
            setattr(self, attr, value)
            self.attrs[attr] = value
        return
########################################################################################################################
class wkf_type():
    def __init__(self):
        return
    def ftc_from_psi(self, psi):
        raise NotImplementedError("The base water retention function should never be "
                                  "actualized check how the class pointer was setup")

    def dftcdpsi_from_psi(self, psi):
        raise NotImplementedError("The base water retention function should never be "
                                  "actualized check how the class pointer was setup")

    def set_wkf_param(self, params_in):
        raise NotImplementedError("The base water retention function should never be "
                                  "actualized check how the class pointer was setup")

class wkf_type_vg(wkf_type):
    def __init__(self):
        super().__init__()
        # # Additional parameters for van Genuchten model
        # self.alpha = 0.0  # Inverse air entry parameter [m^3/MPa]
        # self.n_vg = 0.0   # Pore size distribution parameter
        # self.m_vg = 0.0   # Pore size distribution parameter
        # self.tort = 0.0   # Tortuosity parameter
        # self.th_sat = 0.0  # Saturation volumetric water content [m^3/m^3]
        # self.th_res = 0.0  # Residual volumetric water content [m^3/m^3]

    def ftc_from_psi(self, psi):
        n = self.n_vg
        m = self.m_vg

        psi_eff = -psi
        num = (1.0 - ((self.alpha * psi_eff) ** n / (1.0 + (self.alpha * psi_eff) ** n)) ** m) ** 2.0
        den = (1.0 + (self.alpha * psi_eff) ** n) ** (self.tort * m)

        # van Genuchten 1980 assumes a positive pressure convention
        mask = psi < 0.0
        ftc = torch.where(mask, torch.clamp(num / den, max= 1.0, min = min_ftc), 1.0)
        return ftc

    def dftcdpsi_from_psi(self, psi):
        # Constants
        min_ftc = 1.0e-12

        # Parameters
        n = self.n_vg
        m = self.m_vg

        # Variable
        psi_eff = -psi  # Switch VG 1980 convention

        # Calculate current ftc to see if we are at min
        ftc = self.ftc_from_psi(psi)

        t1  = (self.alpha * psi_eff) ** (n * m)
        dt1 = self.alpha * (n * m) * (self.alpha * psi_eff) ** (n * m - 1.0)

        t2  = (1.0 + (self.alpha * psi_eff) ** n) ** (-m)
        dt2 = -m * (1.0 + (self.alpha * psi_eff) ** n) ** (-m - 1.0) * \
              n * (self.alpha * psi_eff) ** (n - 1.0) * self.alpha

        t3  = (1.0 + (self.alpha * psi_eff) ** n) ** (self.tort * m)
        dt3 = self.tort * m * (1.0 + (self.alpha * psi_eff) ** n) ** (self.tort * m - 1.0) * \
              n * (self.alpha * psi_eff) ** (n - 1.0) * self.alpha

        dftcdpsi = 2.0 * (1.0 - t1 * t2) * (t1 * dt2 + t2 * dt1) / t3 - \
                   t3 ** (-2.0) * dt3 * (1.0 - t1 * t2) ** 2.0

        mask = torch.logical_or(ftc <= min_ftc, psi >= 0.0)
        dftcdpsi = torch.where(mask, 0.0, dftcdpsi)  # We cap ftc, so derivative is zero

        return dftcdpsi

    def set_wkf_param(self, params_in):
        self.alpha  = params_in[0]
        self.n_vg   = params_in[1]
        self.m_vg   = params_in[2]
        self.th_sat = params_in[3]
        self.th_res = params_in[4]
        self.tort   = params_in[5]
        return
########################################################################################################################
class wkf_type_cch(wkf_type):
    def __init__(self):
        super().__init__()
        # self.th_sat = 0.0   # Saturation volumetric water content [m3/m3]
        # self.psi_sat = 0.0  # Bubbling pressure (potential at saturation) [Mpa]
        # self.beta = 0.0     # Clapp-Hornberger "beta" parameter [-]
    def ftc_from_psi(self, psi):
        psi_eff = torch.clamp(psi, max = self.psi_sat)
        ftc     = (psi_eff / self.psi_sat) ** (-2.0 - 3.0 / self.beta)
        return ftc

    def dftcdpsi_from_psi(self, psi):
        mask     = psi < self.psi_sat
        mask     = mask.type(torch.uint8)
        dftcdpsi = torch.where(mask,
                               (-2.0 - 3.0 / self.beta) / self.psi_sat * (psi / self.psi_sat) ** (-3.0 - 3.0 / self.beta)
                               ,0.0)
        return dftcdpsi
    def set_wkf_param(self, params_in):
        self.th_sat  = params_in[0]
        self.psi_sat = params_in[1]
        self.beta    = params_in[2]
        self.attrs = {'th_sat': self.th_sat, 'psi_sat': self.psi_sat, 'beta': self.beta}
        return

    # Method to subset selected attributes and return them as a dictionary
    def subsetAttrs(self, indices):
        return {attr: getattr(self, attr)[indices] for attr in ['th_sat', 'psi_sat', 'beta'] if hasattr(self, attr)}

    def updateAttrs(self, subsetParams):
        for attr, value in subsetParams.items():
            setattr(self, attr, value)
            self.attrs[attr] = value
        return
########################################################################################################################
class wkf_type_smooth_cch(wkf_type):
    def __init__(self):
        super().__init__()
        # self.th_sat = 0.0  # Saturation volumetric water content [m3/m3]
        # self.psi_sat = 0.0  # Bubbling pressure (potential at saturation) [Mpa]
        # self.beta = 0.0  # Clapp-Hornberger "beta" parameter [-]
        # self.scch_pu = 0.0  # An estimated breakpoint capillary pressure, below which the specified water retention curve is applied. It is also the lower limit when the smoothing function is applied. [Mpa]
        # self.scch_ps = 0.0  # An estimated breakpoint capillary pressure, an upper limit where the smoothing function is applied. [Mpa]
        # self.scch_b2 = 0.0  # Constant coefficient of the quadratic term in the smoothing polynomial function [-]
        # self.scch_b3 = 0.0  # Constant coefficient of the cubic term in the smoothing polynomial function [-]

    def ftc_from_psi(self, psi):
        pc         = psi
        sat_res    = 0.0
        alpha      = -1.0 / self.psi_sat
        lambda_ = 1.0 / self.beta

        mask1 = pc <= self.scch_pu
        mask2 = pc < self.scch_ps

        deltaPc = pc - self.scch_ps
        Se      = torch.where(mask1, (-alpha * pc) ** (-lambda_),
                              torch.where(mask2, 1.0 + deltaPc * deltaPc * (self.scch_b2 + deltaPc * self.scch_b3), torch.nan))

        kr      = torch.where(mask1 | mask2,Se ** (3.0 + 2.0 / lambda_), 1.0)

        ftc     = torch.clamp(kr, min = min_ftc)

        return ftc
   

    def dftcdpsi_from_psi(self, psi):
        pc          = psi
        sat_res     = 0.0
        alpha       = -1.0 / self.psi_sat
        lambda_ = 1.0 / self.beta

        mask1 = (pc <= self.scch_pu)
        mask2 = ( pc < self.scch_ps)

        deltaPc = pc - self.scch_ps
        Se      = torch.where(mask1, (-alpha * pc) ** (- lambda_ ),
                         torch.where(mask2,1.0 + deltaPc * deltaPc * (self.scch_b2 + deltaPc * self.scch_b3) ,torch.nan))

        kr      = torch.where(mask1, Se ** (3.0 + 2.0 / lambda_ ),
                              torch.where(mask2,Se ** (2.50 + 2.0 / lambda_ ), 1.0 ))

        dSe_dpc = torch.where(mask1, - lambda_ * Se / pc,
                              torch.where(mask2, deltaPc * (2 * self.scch_b2 + 3 * deltaPc * self.scch_b3), torch.nan))

        dkr_dSe = torch.where(mask1, (3.0 + 2.0 / lambda_) * kr / Se,
                              torch.where(mask2, (2.50 + 2.0 / lambda_ ) * kr / Se, torch.nan ))

        dkr_dp  = torch.where(mask1 | mask2, dkr_dSe * dSe_dpc, 0.0)

        dftcdpsi = dkr_dp
        dftcdpsi = torch.where((kr <= min_ftc), 0.0, dftcdpsi)
        return dftcdpsi


    def set_wkf_param(self, params_in):
        self.th_sat  = params_in[0]
        self.psi_sat = params_in[1]
        self.beta    = params_in[2]
        styp         = int(params_in[3]) #! an option to force constant coefficient of the
                                         #! quadratic term 0 (styp = 1) or to force the constant
                                         #! coefficient of the cubic term 0 (styp/=2)

        alpha        = -1.0 / self.psi_sat
        lambda_val   = 1.0  / self.beta
        ps           = -0.9 / alpha
        self.scch_ps = ps

        if styp == 1:
            # Choose `pu` that forces `scch_b2 = 0`.
            pu = findGu_SBC_zeroCoeff(lambda_val, torch.tensor([3]), -alpha * ps) / (-alpha)
            self.scch_pu = pu

            # Find helper constants.
            bcAtPu            = (-alpha * pu) ** (-lambda_val)
            lambdaDeltaPuOnPu = lambda_val * (1.0 - ps / pu)
            oneOnDeltaPu      = 1.0 / (pu - ps)

            # Store coefficients for cubic function.
            self.scch_b2 = 0.0
            self.scch_b3 = (2.0 - bcAtPu * (2.0 + lambdaDeltaPuOnPu)) * oneOnDeltaPu ** 3
            if torch.any(self.scch_b3 <= 0.0):
                mask = self.scch_b3 <= 0.0
                print('set_wrf_param_smooth_cch b3 <= 0', pu[mask], ps[mask], alpha[mask], lambda_val[mask], oneOnDeltaPu[mask],
                      lambdaDeltaPuOnPu[mask], bcAtPu[mask], self.psi_sat[mask])
                raise ValueError('Invalid value for scch_b3')
        else:
            # Choose `pu` that forces `sbc_b3 = 0`.
            pu = findGu_SBC_zeroCoeff(lambda_val, torch.tensot([2]), -alpha * ps) / (-alpha)
            self.scch_pu = pu

            # Find helper constants.
            bcAtPu            = (-alpha * pu) ** (-lambda_val)
            lambdaDeltaPuOnPu = lambda_val * (1.0 - ps / pu)
            oneOnDeltaPu      = 1.0 / (pu - ps)

            # Store coefficients for cubic function.
            self.scch_b2 = -(3.0 - bcAtPu * (3.0 + lambdaDeltaPuOnPu)) * oneOnDeltaPu ** 2
            if self.scch_b2 >= 0.0:
                print('set_wrf_param_smooth_cch b2 <= 0')
                raise ValueError('Invalid value for scch_b2')
            self.scch_b3 = 0.0
        
########################################################################################################################
class wkf_type_tfs(wkf_type):
    def __init__(self):
        super().__init__()
        # self.p50 = 0.0
        # self.avuln = 0.0
        # self.th_sat = 0.0

    def ftc_from_psi(self, psi):
        psi_eff = torch.clamp(psi, max = -nearzero)
        ftc     = torch.clamp( 1.0 / (1.0 + (psi_eff / self.p50) ** self.avuln), min = min_ftc)
        return ftc

    def dftcdpsi_from_psi(self, psi):
        ftc  = 1.0 / (1.0 + (psi / self.p50) ** self.avuln)
        fx   = 1.0 + (psi / self.p50) ** self.avuln
        dfx  = self.avuln * (psi / self.p50) ** (self.avuln - 1.0) * (1.0 / self.p50)
        dftcdpsi = -fx ** (-2.0) * dfx

        mask     =  ftc < min_ftc
        dftcdpsi =  torch.where(mask, 0.0, dftcdpsi)

        mask     =  psi > 0.0
        dftcdpsi =  torch.where(mask, 0.0, dftcdpsi)

        return dftcdpsi

    def set_wkf_param(self, params_in):
        self.p50   = params_in[0]
        self.avuln = params_in[1]
        self.attrs = {'p50': self.p50, 'avuln': self.avuln}
        return

    # Method to subset selected attributes and return them as a dictionary
    def subsetAttrs(self, indices):
        return {attr: getattr(self, attr)[indices] for attr in ['p50', 'avuln'] if hasattr(self, attr)}

    def updateAttrs(self, subsetParams):
        for attr, value in subsetParams.items():
            setattr(self, attr, value)
            self.attrs[attr] = value
        return
########################################################################################################################
"""
Define some separate functions
"""
def findGu_SBC_zeroCoeff(lambda_, AA, gs, relTol=1e-12, max_iter=10000):

    # Check arguments
    """
    Function to find the root using bracketed Newton-Raphson method.
    """
    if torch.any(lambda_ <= 0) or torch.any(lambda_ >= 2) or \
            torch.any((AA != 2) & (AA != 3)) or \
            torch.any(gs >= 1) or torch.any(gs < 0):
        raise ValueError("findGu_SBC_zeroCoeff: bad parameters")

    gu = (AA / (AA + lambda_)) ** (-1.0 / lambda_)
    gu[gs <= 0] = 1.0  # If gs is 0, solution is trivial

    for i in range(max_iter):
        guInv = 1.0 / gu
        guToMinusLam = gu ** (-lambda_)
        gsOnGu = gs * guInv
        resid = AA - guToMinusLam * (AA + lambda_ - lambda_ * gsOnGu)

        dr_dGu = (1.0 + lambda_) * (1.0 - gsOnGu) + (AA - 1)
        dr_dGu = lambda_ * guToMinusLam * guInv * dr_dGu

        deltaGu = resid / dr_dGu
        gu = gu - deltaGu

        if torch.max(torch.abs(deltaGu)) < relTol:
            break

    if i == max_iter - 1:
        print("Warning: Maximum iterations reached in findGu_SBC_zeroCoeff")

    return gu
########################################################################################################################
def solutepsi(th,rwc_ft,th_sat,th_res,pinot):
    # !
    # ! !DESCRIPTION: computes solute water potential (negative) as a function of
    # !  water content for the plant PV curve.
    # !
    # ! !USES:
    # !
    # ! !ARGUMENTS
    #
    # real(r8)      , intent(in)     :: th          ! vol wc       [m3 m-3]
    # real(r8)      , intent(in)     :: rwc_ft
    # real(r8)      , intent(in)     :: th_sat
    # real(r8)      , intent(in)     :: th_res
    # real(r8)      , intent(in)     :: pinot
    # real(r8)      , intent(out)    :: psi         ! water potential   [MPa]
    #
    # ! -----------------------------------------------------------------------------------
    # ! From eq 8, Christopherson et al:
    # !
    # ! psi = pinot/RWC*, where RWC*=(rwc-rwc_res)/(rwc_ft-rwc_res)
    # ! psi = pinot * (rwc_ft-rwc_res)/(rwc-rwc_res)
    # !
    # ! if rwc_res =  th_res/th_sat
    # !
    # !     = pinot * (rwc_ft - th_res/th_sat)/(th/th_sat - th_res/th_sat )
    # !     = pinot * (th_sat*rwc_ft - th_res)/(th - th_res)
    # ! -----------------------------------------------------------------------------------

    psi = pinot * (th_sat*rwc_ft - th_res) / (th - th_res)

    return psi
##############################################################################################################
def pressurepsi(th,rwc_ft,th_sat,th_res,pinot,epsil):
    # !
    # ! !DESCRIPTION: computes pressure water potential (positive) as a function of
    # !  water content for the plant PV curve.
    # !
    # ! !USES:
    # !
    # ! !ARGUMENTS
    # real(r8) , intent(in)  :: th
    # real(r8) , intent(in)  :: rwc_ft
    # real(r8) , intent(in)  :: th_sat
    # real(r8) , intent(in)  :: th_res
    # real(r8) , intent(in)  :: pinot
    # real(r8) , intent(in)  :: epsil
    # real(r8) , intent(out) :: psi         ! water potential   [MPa]

    psi = epsil * (th - th_sat*rwc_ft) / (th_sat*rwc_ft-th_res) - pinot

    return psi
##############################################################################################################
def capillarypsi(th,th_sat,cap_int,cap_slp):
    # !
    # ! !DESCRIPTION: computes water potential in the capillary region of the plant
    # !  PV curve (sapwood only)
    # !
    # ! !ARGUMENTS
    #
    # real(r8)      , intent(in)     :: th          ! water content     [m3 m-3]
    # real(r8)      , intent(in)     :: th_sat
    # real(r8)      , intent(in)     :: cap_int
    # real(r8)      , intent(in)     :: cap_slp
    # real(r8)      , intent(out)    :: psi         ! water potential   [MPa]

    psi = cap_int + th*cap_slp/th_sat

    return psi
##############################################################################################################
def dsolutepsidth(th,th_sat,th_res,rwc_ft,pinot):

    # !
    # ! !DESCRIPTION: returns derivative of solutepsi() wrt theta
    # !
    # ! !USES:
    # !
    # ! !ARGUMENTS
    # real(r8)      , intent(in)     :: th
    # real(r8)      , intent(in)     :: th_sat
    # real(r8)      , intent(in)     :: th_res
    # real(r8)      , intent(in)     :: rwc_ft
    # real(r8)      , intent(in)     :: pinot
    # real(r8)      , intent(out)    :: dpsi_dth
    #
    # ! -----------------------------------------------------------------------------------
    # ! Take derivative of eq 8 (function solutepsi)
    # ! psi      =  pinot * (th_sat*rwc_ft - th_res) * (th - th_res)^-1
    # ! dpsi_dth = -pinot * (th_sat*rwc_ft - th_res) * (th - th_res)^-2
    # ! -----------------------------------------------------------------------------------

    dpsi_dth = -1.0 * pinot*(th_sat * rwc_ft - th_res )*(th - th_res)**(-2.0)

    return dpsi_dth
##############################################################################################################
def dpressurepsidth(th_sat,th_res,rwc_ft,epsil):
    # !
    # ! !DESCRIPTION: returns derivative of pressurepsi() wrt theta
    # !
    # ! !USES:
    # !
    # ! !ARGUMENTS
    # real(r8)      , intent(in)     :: th_sat
    # real(r8)      , intent(in)     :: th_res
    # real(r8)      , intent(in)     :: rwc_ft
    # real(r8)      , intent(in)     :: epsil
    # real(r8)      , intent(out)    :: dpsi_dth       ! derivative of water potential wrt theta  [MPa m3 m-3]

    dpsi_dth = epsil/(th_sat*rwc_ft - th_res)

    return dpsi_dth
##############################################################################################################
def dcapillarypsidth(cap_slp,th_sat):
    # !
    # ! !DESCRIPTION: returns derivative of capillaryPV() wrt theta
    # !
    # ! !USES:
    # !
    # ! !ARGUMENTS
    # real(r8)       , intent(in)     :: cap_slp
    # real(r8)       , intent(in)     :: th_sat
    # real(r8)      , intent(out)    :: y           ! derivative of water potential wrt theta  [MPa m3 m-3]

    y = cap_slp/th_sat

    return y
##############################################################################################################
# class wrf_arr_type:
#     def __init__(self):
#         self.p = wrf_type()
#
# class wkf_arr_type:
#     def __init__(self):
#         self.p = wkf_type()

##################################################################################################################
