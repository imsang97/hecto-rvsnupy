#!/usr/bin/env python
# coding: utf-8

"""
RVSNUpy Run: for ALL fibers in a single observation date & config

Python script to measure redshifts of ALL 300 MMT/Hectospec fibers using RVSNUpy,
for a single observation date specified in User Settings below.
This script assumes that the raw data are reduced with the 'HSRED' pipeline.
This script is designed especially for MMT/Hectospec observations of 2024A, 2025B,
and 2026A semester, by Sang Hyeok Im (sanghyeok.im97@gmail.com).

Saves one .dat file per config next to the spHect*.fits files:
  MMTreduction/{obs_date}/reduction/{rerun}/zResults_{config_str}.dat

Skips any config whose output file already exists, allowing the script to be
safely interrupted and resumed.

Paths to the reduced data
- pwd = /data1/imsang/Research/ETG_satellite/MMTreduction/YYYY.MMDD/

Path to the RVSNUpy
- /data1/imsang/Research/ETG_satellite/RVSNUpy/src/
"""

import sys
import glob, os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u

from specutils import Spectrum1D
from specutils import manipulation

from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()


## --- For RVSNUpy --- ##
sys.path.append('/data1/imsang/Research/ETG_satellite/RVSNUpy/src/')
os.environ['rvsnupy'] = f"/data1/imsang/Research/ETG_satellite/RVSNUpy"

import RVSNUpy
from RVSNUpy.template import sdss_galaxy_templates, syn_abstemplates, syn_emtemplates
from RVSNUpy.template import zp_calib
from RVSNUpy.rvm import rvm
## ------------------- ##


## --- my tools --- ##
sys.path.append('/data1/imsang/Spyder_Projects/my_modules')
from my_visualize import visualize as vi
import my_matplotlib
my_matplotlib.import_my_plt_settings()

sys.path.append('/data1/imsang/Research/ETG_satellite/notebooks/3_get_redshifts')
import z_snupy_tools as sptools
## ------------------- ##




## ========= User Settings ========= ##
n_jobs    = 4                  ## number of CPUs to use
chi_thres = np.inf             ## maximum chi2 value (np.inf = keep all finite results)
r_thres   = 0                  ## minimum r value (0 = keep all positive r results)
spectrum_range = [3800, 9200]  ## wavelength range for redshift measurement (in Angstrom)

## target obs_date and cfg ##
obs_date = '2024.0410'
rerun    = '0100'

## Gaussian smoothing stddev: for visualization only ##
stddev = 3  # in [pixels]

## --- Telluric mask --- ##
## NOTE: do NOT add a 'Mask':[9200,9250] entry here. The red-end cutoff is
## handled by spectrum_range=[3800, 9200] in z_single. Adding a mask entry
## for [9200,9250] causes a Schoenberg-Whitney violation in splrep for fibers
## whose BPM also masks that tail -> all templates return NaN (see CASE 5 in
## zForClaude/why_nan_for_all_templates.txt).
telluric_lines = {
        r'$[OI]$'   : [5555.0, 5595.0],
        r'$O_2$ A'  : [7594.0-10, 7621.0+10],
        r'$O_2$ B'  : [6867.0, 6884.0],
        r'$H_2O$ 1' : [7186.0, 7270.0],
        r'$H_2O$ 2' : [8210.0, 8240.0],
    }
mask = []
for key in telluric_lines.keys():
    mask.append(telluric_lines[key])

## ================================= ##


print(f"\n === Start of the Program === ")
print(f' * n_jobs         = {n_jobs}')
print(f' * chi_thres      = {chi_thres}')
print(f' * r_thres        = {r_thres}')
print(f' * spectrum_range = {spectrum_range} Angstrom\n')


stime = time.time()


## ======== Import templates and initialize ======== ##

print(f"\n --- Import Galaxy Templates --- ")

## --- Import Galaxy Templates --- ##
gal_temps     = sdss_galaxy_templates("air")
syn_abs_temps = syn_abstemplates("air")
syn_em_temps  = syn_emtemplates("air")

# +) "Galaxy_air" template
temp_pwd   = f"/data1/imsang/Research/ETG_satellite/RVSNUpy/template_files/csv/sdss/"
temp_fname = f"Galaxy_air.csv"

temp_data = pd.read_csv(f"{temp_pwd}{temp_fname}")
temp_data['mask'] = np.full(len(temp_data), 1.)
temp_data = np.array(temp_data).T
temp_to_add = {"Galaxy_air": [temp_data, 2]}


## --- rename some of them --- ##
keys_maps = {'3Gyr': 'syn_abs_3Gyr',
             '5Gyr': 'syn_abs_5Gyr',
             '7Gyr': 'syn_abs_7Gyr',
             '9Gyr': 'syn_abs_9Gyr',
             '11Gyr': 'syn_abs_11Gyr'}
for old_key, new_key in keys_maps.items():
    if old_key in syn_abs_temps:
        syn_abs_temps[new_key] = syn_abs_temps.pop(old_key)

syn_em_temps['syn_em_0.01Gyr'] = syn_em_temps.pop('0.01Gyr')


## --- concat all templates --- ##
all_temps = gal_temps | syn_abs_temps | syn_em_temps | temp_to_add

print(f" * Templates loaded. N_templates = {len(all_temps)}")

## ================================================= ##




## ====================================================== ##
## ====  Single date: read & measure  ================== ##
## ====================================================== ##

year = obs_date[:4]

print(f"\n\n")
print(f" ######################################### ")
print(f" ## obs date = {obs_date},  rerun = {rerun} ##")
print(f" ######################################### ")
print(f"")

reduct_pwd = f"/data1/imsang/Research/ETG_satellite/MMTreduction/{obs_date}/reduction/{rerun}/"
sav_pwd    = reduct_pwd   ## save results next to the spHect*.fits files


## -------------------------------------------------- ##
## ----  1 - (1) Find the "spHect*fits" files  ------ ##
## -------------------------------------------------- ##

spHect_fname = "spHect-NGC*.fits"
spHect_flist = sorted(glob.glob(reduct_pwd + spHect_fname))

config_strs = []
cat_fnames  = []

for i in range(len(spHect_flist)):
    print(f" * spHect_flist[{i}]  : {spHect_flist[i][len(reduct_pwd):]}")
    if year == '2024' or year == '2026':
        config_str = spHect_flist[i][len(reduct_pwd):][7:14] + '-' + spHect_flist[i][len(reduct_pwd):][36:37]
        cat_fname  = spHect_flist[i][len(reduct_pwd):][7:35] + '.cat'
    elif year == '2025':
        config_str = spHect_flist[i][len(reduct_pwd):][7:14] + '-' + spHect_flist[i][len(reduct_pwd):][40:41]
        cat_fname  = spHect_flist[i][len(reduct_pwd):][7:39] + '.cat'
    config_strs.append(config_str)
    cat_fnames.append(cat_fname)

if len(spHect_flist) == 0:
    print(f" ! No spHect-NGC*.fits files found in {reduct_pwd}.")
    print(f" ! Exiting.")
    sys.exit(1)


## -------------------------------------------------- ##
## ----  1 - (2) Read spectra from the files  ------- ##
## -------------------------------------------------- ##

print(f"\n --- Read spHect*.fits Files --- ")
print(f" * observation date : {obs_date}")
print(f" * rerun            : {rerun}")
print(f" * pwd              : {reduct_pwd}")
print(f"  => N_files = {len(spHect_flist)} \n ")

spectrum_lists = []

for i in range(len(spHect_flist)):
    print(f" - cfg = {config_strs[i]} ")
    print(f"    * file name : {spHect_flist[i][len(reduct_pwd):]} ")

    with fits.open(spHect_flist[i]) as sphect_hdul:
        wavelength_list = sphect_hdul[0].data
        counts_list     = sphect_hdul[1].data
        variance_list   = 1 / sphect_hdul[2].data
        bpm_list        = sphect_hdul[4].data

        std_list  = np.sqrt(variance_list)
        mask_list = (bpm_list * -1) + 1

        ## --- Mask NaN values --- ##
        nanCount = np.isnan(counts_list)
        nanVar   = np.isnan(variance_list) | (variance_list == 0)
        nanMask  = nanCount | nanVar
        mask_list = mask_list.astype(bool) & ~nanMask
        mask_list = mask_list.astype(np.int32)
        ## ---------------------- ##

        spectrum_list = np.stack([wavelength_list, counts_list, std_list, mask_list], axis=1)
        spectrum_lists.append(spectrum_list)

    print(f"    * {spectrum_list.shape[0]} spectra found => spectrum_lists[{i}] ")


## -------------------------------------------------- ##
## ----  1 - (3) Read fiber assignment table  ------- ##
## -------------------------------------------------- ##

fibTabFname = "spHect-NGC*.csv"
fibTabFlist = sorted(glob.glob(reduct_pwd + fibTabFname))

print(f"\n --- Reading spHect*_FibInfo_catID_added.csv Files --- ")
print(f" * pwd             : {reduct_pwd}")

fibTabList = []
for i in range(len(fibTabFlist)):
    print(f"  - fibTabFlist[{i}] : {fibTabFlist[i][len(reduct_pwd):]}")
    fibTab = ascii.read(fibTabFlist[i])
    fibTabList.append(fibTab)


## -------------------------------------------------- ##
## ----  1 - (4) Calculate Gaussian-smoothed Spectra  ##
## -------------------------------------------------- ##
## PASS; it is only for visualization.

# gs_flux_lists = []

# for i in range(len(spectrum_lists)):
#     spectrum_list = spectrum_lists[i]
#     gs_flux_list  = []

#     for j in range(len(spectrum_list)):
#         flux = spectrum_list[j, 1]
#         wav  = spectrum_list[j, 0]

#         spec1d = Spectrum1D(flux=flux*u.Jy, spectral_axis=wav*u.Angstrom)
#         spec1d = manipulation.gaussian_smooth(spectrum=spec1d, stddev=stddev)

#         flux_gs = spec1d.flux.value
#         gs_flux_list.append(flux_gs)

#     gs_flux_list = np.array(gs_flux_list)
#     gs_flux_lists.append(gs_flux_list)


## ====================================================== ##
## ====  Loop over all configs in this date  ============ ##
## ====================================================== ##

## --- Initialize RVSNUpy once (shift_templates is expensive) --- ##
measure = rvm(all_temps, n_jobs=n_jobs)
## -------------------------------------------------------------- ##

for cfg_idx in range(len(spectrum_lists)):

    config_str = config_strs[cfg_idx]
    fibTab     = fibTabList[cfg_idx]
    n_fibers   = len(fibTab)

    print(f"\n\n --- Config: {config_str} ({n_fibers} fibers) --- ")

    ## --- Skip if result file already exists --- ##
    sav_fname = f"zResults_{config_str}.dat"
    if os.path.exists(sav_pwd + sav_fname):
        print(f" * Result file already exists. Skipping {config_str}.")
        print(f"   ({sav_pwd + sav_fname})")
        continue

    ## ------------------------------------------ ##
    ## ----  Measure redshifts for all fibers  --- ##
    ## ------------------------------------------ ##

    print(f"\n --- Measuring redshifts for all {n_fibers} fibers --- ")
    print(f"  {'fib_idx':^10}  {'catID':^22}  {'OBJTYPE':^20}  {'N_measurements':^20}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*20}  {'-'*20}")

    rows = []

    for j in range(n_fibers):

        catID   = str(fibTab['catID'][j]).strip()
        objtype = str(fibTab['OBJTYPE'][j]).strip()
        ra      = float(fibTab['RA'][j])
        dec     = float(fibTab['DEC'][j])
        fiberid = fibTab['FIBERID'][j]

        fiber_meta = {
            'obs_date': obs_date,
            'cfg_str' : config_str,
            'fib_idx' : j,
            'FIBERID' : fiberid,
            'catID'   : catID,
            'RA'      : ra,
            'DEC'     : dec,
            'OBJTYPE' : objtype,
        }

        if objtype != 'TARGET':
            print(f"  {j:<10}  {catID:<22}  {objtype:<20}  {'-':^20}")
            rows.append({**fiber_meta,
                         'template_RV' : np.nan,
                         'z_RV'        : np.nan,
                         'zerr_RV'     : np.nan,
                         'r_RV'        : np.nan,
                         'chi_eff_RV'  : np.nan,
                         'best_RV'     : np.nan,
                         'note_RV'     : np.nan,
                         })
            continue

        ## --- Measure redshift --- ##
        df      = measure.z_single(spectrum_lists[cfg_idx][j], mask=mask,
                                   chi_thres=chi_thres, r_thres=r_thres,
                                   spectrum_range=spectrum_range)
        ## ----------------------- ##

        if len(df) >= 1:
            for _, row in df.iterrows():
                rows.append({**fiber_meta,
                             'template_RV' : row['template_name'],
                             'z_RV'        : row['z'],
                             'zerr_RV'     : row['zerr'],
                             'r_RV'        : row['r'],
                             'chi_eff_RV'  : row['chi_eff'],
                             'best_RV'     : row['best'],
                             'note_RV'     : row['note'],
                             })
        else:
            rows.append({**fiber_meta,
                         'template_RV' : np.nan,
                         'z_RV'        : np.nan,
                         'zerr_RV'     : np.nan,
                         'r_RV'        : np.nan,
                         'chi_eff_RV'  : np.nan,
                         'best_RV'     : np.nan,
                         'note_RV'     : np.nan,
                         })

        n_temps = len(df)
        print(f"  {j:<10}  {catID:<22}  {objtype:<20}  {n_temps:<20}")

    ## ------------------------------------------ ##
    ## ----  Save result table for this config --- ##
    ## ------------------------------------------ ##

    print(f"\n --- Save Results: {config_str} --- ")
    print(f" * sav pwd : {sav_pwd}")

    result_df  = pd.DataFrame(rows)
    result_cat = Table.from_pandas(result_df)

    ascii.write(result_cat, sav_pwd + sav_fname, format='fixed_width_two_line', overwrite=True)
    print(f" -> Saved! ({sav_fname})")
    print(f"    * N_fibers = {len(result_df)}")


etime = time.time()

print(f"\n === Done! === ")
print(f" * Total time: {etime - stime:.2f} seconds")
