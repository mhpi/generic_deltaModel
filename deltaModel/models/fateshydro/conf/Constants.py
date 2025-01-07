import torch
# Kinds
fates_r8                = torch.float64                # torch.float64   #! 8 byte real
fates_int               = torch.int64                  # torch.int64     #! 4 byte int
fates_unset_int         = -9999                     # Used to initialize and test unset integers
fates_unset_r8          = -1.0e36                   # Used to initialize and test unset r8s
fates_check_param_set   = 9.9e32                    # Used to check if a parameter was specified in the parameter file
itrue                   = 1                         # Integer equivalent of true and false
ifalse                  = 0
tinyr8                  = torch.finfo(fates_r8).tiny   # torch.finfo(fates_r8).tiny
nearzero                = 1.0e-30
fates_huge              = torch.finfo(fates_r8).max    # torch.finfo(fates_r8).max
fates_tiny              = torch.finfo(fates_r8).tiny   # torch.finfo(fates_r8).tiny
prec                    = 1e-8                      # Avoid zeros to avoid Nans
max_wb_step_err         = 2.0e-6
wind_fill               = 1.0e-3



# Unit conversion constants
umolC_to_kgC    = 12.0E-9                           #! Conversion factor umols of Carbon -> kg of Carbon (1 mol = 12g)
mg_per_kg       = 1.0e6                             #! Conversion factor: miligrams per kilogram
g_per_kg        = 1000.0                            #! Conversion factor: grams per kilograms
kg_per_g        = 0.001                             #! Conversion factor: kilograms per gram
mg_per_g        = 1000.0                            #! Conversion factor: miligrams per grams
kg_per_Megag    = 1000.0                            #! Conversion factor: kilograms per Megagram
umol_per_mmol   = 1000.0                            #! Conversion factor: micromoles per milimole
mmol_per_mol    = 1000.0                            #! Conversion factor: milimoles per mole
umol_per_mol    = 1.0E6                             #! Conversion factor: micromoles per mole
mol_per_umol    = 1.0E-6                            #! Conversion factor: moles per micromole
umol_per_kmol   = 1.0E9                             #! Conversion factor: umols per kilomole
m_per_mm        = 1.0E-3                            #! Conversion factor: meters per milimeter
mm_per_m        = 1.0E3                             #! Conversion factor: milimeters per meter
mm_per_cm       = 10.0                              #! Conversion factor: millimeters per centimeter
m_per_cm        = 1.0E-2                            #! Conversion factor: meters per centimeter
m2_per_ha       = 1.0e4                             #! Conversion factor: m2 per ha
m2_per_km2      = 1.0e6                             #! Conversion factor: m2 per km2
cm2_per_m2      = 10000.0                           #! Conversion factor: cm2 per m2
m3_per_mm3      = 1.0E-9                            #! Conversion factor: m3 per mm3
m3_per_cm3      = 1.0E-6
cm3_per_m3      = 1.0E6                             #! Conversion factor: cubic meters per cubic cm
ha_per_m2       = 1.0e-4                            #! Conversion factor :: ha per m2
sec_per_min     = 60.0                              #! Conversion: seconds per minute
sec_per_day     = 86400.0                           #! Conversion: seconds per day
days_per_sec    = 1.0 / 86400.0                     #! Conversion: days per second
days_per_year   = 365.00                            #! Conversion: days per year. assume HLM uses 365 day calendar.
ndays_per_year  = int(days_per_year)                # Integer version of days per year.
years_per_day   = 1.0 / 365.00                      #! Conversion: years per day. assume HLM uses 365 day calendar.
months_per_year = 12.0                              #! Conversion: months per year
J_per_kJ        = 1000.0                            #! Conversion: Joules per kiloJoules
megajoules_per_joule = 1.0E-6                       #! Conversion: megajoules per joule
photon_to_e     = 0.5
wm2_to_umolm2s  = 4.6

# Physical constants
rgas_J_K_kmol           = 8314.4598                 #! universal gas constant [J/K/kmol]
rgas_J_K_mol            = 8.3144598                 #! universal gas constant [J/k/mol]
t_water_freeze_k_1atm   = 273.15                    #! freezing point of water at 1 atm (K)
t_water_freeze_k_triple = 273.16                    #! freezing point of water at triple point (K)
denice                  = 0.917e3                   # ! density of ice
Rda                     = 287.05                    # is the gas constant for dry air (J kg-1 K-1)
dens_fresh_liquid_water = 1.0E3                     #! Density of fresh liquid water (kg/m3)
molar_mass_water        = 18.0                      #! Molar mass of water (g/mol)
molar_mass_ratio_vapdry = 0.622                     #! Approximate molar mass of water vapor to dry air (-)
grav_earth              = 9.8                       #! Gravity constant on earth [m/s]
pa_per_mpa              = 1.0e6                     #! Megapascals to pascals
mpa_per_pa              = 1.0e-6                    #! Pascals to megapascals
mpa_per_mm_suction      = dens_fresh_liquid_water * grav_earth * 1.0E-9  #! Conversion: megapascals per mm H2O suction
area                    = 10000.0
area_inv                = 1.0e-4

# Geometric Constants
pi_const = torch.pi

min_max_dbh_for_trees   = 15.0  #! If pfts have a max dbh less
                                #! than this value FATES
                                #! will use the default regeneration scheme.
                                #! Avoids TRS for shrubs / grasses.
#===========================================================================================================
psi_aroot_init              = -0.2
dh_dz                       = 0.02
fates_hydro_psi0            = 0
fates_hydro_psicap          = -0.6
hydr_kmax_rsurf1            = 20
zsapric                     = 0.5
pcalpha                     = 0.5
pcbeta                      = 0.139
min_trim                    = 0.1
error_thresh                = 1.0e-5
c3_path_index               = 1

kcha                        = 79430.0
koha                        = 36380.0
cpha                        = 37830.0
lmrha                       = 46390.0
lmrhd                       = 150650.0
lmrse                       = 490.0
lmrc                        = 1.15912391
s_fix                       = -6.25
a_fix                       = -3.62
b_fix                       = 0.27
c_fix                       = 25.15
fnps                        = 0.15
theta_psii                  = 0.7
theta_cj_c3                 = 0.999
theta_cj_c4                 = 0.999
theta_ip                    = 0.999

init_a2l_co2_c3             = 0.7
init_a2l_co2_c4             = 0.4
rsmax0                      = 2.0e8
quant_eff                   = 0.05
dleaf                       = 0.04
Cb                          = 1.78
Nlc                         = 0.25
Fabs                        = 0.85
Fref                        = 0.07
kmax_petiole_to_leaf        = 1.0e8
fine_root_radius_const=rs1  = 0.0001
hydr_kmax_rsurf2            = 0.0001
large_kmax_bound            = 1.0e4
mm_kc25_umol_per_mol        = 404.9
mm_ko25_mmol_per_mol        = 278.4
co2_cpoint_umol_per_mol     = 42.75
h2o_co2_bl_diffuse_ratio    = 1.4
h2o_co2_stoma_diffuse_ratio = 1.6

lmr_b                       = 0.1012    #! (degrees C**-1)
lmr_c                       = -0.0005   #! (degrees C**-2)
lmr_TrefC                   = 25.0    #! (degrees C)
lmr_r_1                     = 0.2061    #! (umol CO2/m**2/s / (gN/(m2 leaf)))
lmr_r_2                     = -0.0402   #! (umol CO2/m**2/s/degree C)


frag_cwd_frac = [0.045, 0.075, 0.21, 0.67]
branch_frac   = sum(frag_cwd_frac[:3])

# //    "frag_cwd_frac": {
# //
# //        "leaf_p_media" :    0.045,
# //        "stem_p_media" :    0.075,
# //        "troot_p_media":    0.21,
# //        "aroot_p_media":    0.67
# //
# //    },
dz   = [0.10, 0.20 , 0.30, 0.4, 1.0]
zi   = [0.10, 0.3, 0.6, 1.0, 2.0]
zsoi = [0.05, 0.20, 0.45, 0.80, 1.5]

