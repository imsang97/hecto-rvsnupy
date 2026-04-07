#!/usr/bin/env python
# coding: utf-8

"""
RVSNUpy Batch Run: for galaxies with known z_spec

Python script to measure redshifts of MMT/Hectospec spectra using RVSNUpy.
This script assumes that the raw data are reduced with the 'HSRED' pipeline.
This script is designed especially for MMT/Hectospec observations of 2024A, 2025B,
and 2026A semester, by Sang Hyeok Im (sanghyeok.im97@gmail.com).

Runs over ALL observation dates (2024.0410 - 2026.0221) and collects all
redshift measurements for spec-z-known galaxies into a single output file.

Paths to the reduced data
- pwd = /data1/imsang/Research/ETG_satellite/MMTreduction/YYYY.MMDD/

Path to the RVSNUpy
- /data1/imsang/Research/ETG_satellite/RVSNUpy/src/
"""

import sys
import glob, os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import patches

from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.io import ascii
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.utils import data

from specutils import Spectrum1D
from specutils import manipulation

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
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
spectrum_range = [3800, 9200]  ## wavelength range to consider for redshift measurement (in Angstrom)

obs_date_rerun_list = [
    ('2024.0410', '0100'),
    ('2024.0411', '0100'),
    ('2024.0412', '0100'),
    ('2024.0413', '0100'),
    ('2024.0415', '0100'),
    ('2024.0416', '0100'),
    ('2024.0611', '0100'),
    ('2024.0612', '0100'),
    ('2024.0613', '0100'),
    ('2025.0929', '0100'),
    ('2025.1002', '0100'),
    ('2025.1003', '0100'),
    ('2026.0221', '0100'),
]

notebook_dir = f"/data1/imsang/Research/ETG_satellite/notebooks/3_get_redshifts/"
sav_pwd_figs = notebook_dir + f"redshift_measurements/knownSpecZ/figs/"
sav_pwd_dat  = notebook_dir + f"redshift_measurements/knownSpecZ/"

## --- Gaussian smoothing stddev --- ##
stddev = 3  # in [pixels]

## --- Telluric mask --- ##
telluric_lines = {
        r'$[OI]$': [5555.0, 5595.0],
        r'$O_2$ A': [7594.0-10, 7621.0+10],
        r'$O_2$ B': [6867.0, 6884.0],
        r'$H_2O$ 1': [7186.0, 7270.0],
        r'$H_2O$ 2': [8210.0, 8240.0],
        'Mask': [9200, 9250]
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







## ======== Import templates and initialize ======== ##

print(f"\n --- Import Galaxy Templates --- ")

## --- Import Galaxy Templates --- ##
gal_temps     = sdss_galaxy_templates("air")
syn_abs_temps = syn_abstemplates("air")
syn_em_temps  = syn_emtemplates("air")

# +) "Galaxy_air" template..
temp_pwd   = f"/data1/imsang/Research/ETG_satellite/RVSNUpy/template_files/csv/sdss/"
temp_fname = f"Galaxy_air.csv"

temp_data = pd.read_csv(f"{temp_pwd}{temp_fname}")
temp_data['mask'] = np.full(len(temp_data), 1.)
temp_data = np.array(temp_data).T
temp_to_add = {"Galaxy_air": [temp_data, 2]}


## --- rename some of them.. --- ##
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

measure = rvm(all_temps, n_jobs=4)

print(f" * Templates loaded. N_templates = {len(all_temps)}")

## ================================================= ##




## --- Make save directories if not exist --- ##
if not os.path.exists(sav_pwd_figs):
    os.makedirs(sav_pwd_figs)


## ====================================================== ##
## ====  Main Loop: over all observation dates  ========= ##
## ====================================================== ##


all_SDSS_zs = []
all_best_zs = []
all_specz_gals_withResults = []

for date, rerun in obs_date_rerun_list:

    year = date[:4]

    print(f"\n\n")
    print(f" ######################################### ")
    print(f" ## obs date = {date},  rerun = {rerun} ##")
    print(f" ######################################### ")
    print(f"")
    


    ## -------------------------------------------------- ##
    ## ----  1 - (1) Find the "spHect*fits" files  ------ ##
    ## -------------------------------------------------- ##

    reduct_pwd   = f"/data1/imsang/Research/ETG_satellite/MMTreduction/{date}/reduction/{rerun}/"
    spHect_fname = "spHect-NGC*.fits"
    spHect_flist = sorted(glob.glob(reduct_pwd + spHect_fname))

    config_strs = []
    cat_fnames  = []

    # print(f"\n = = = Reading spHect*.fits Files = = = ")
    # print(f" * pwd              : {reduct_pwd}")
    for i in range(len(spHect_flist)):
        print(f" * spHect_flist[{i}]  : {spHect_flist[i][len(reduct_pwd):]}")
        if year == '2024' or year == '2026':
            config_str = spHect_flist[i][len(reduct_pwd):][7:14] + '-' + spHect_flist[i][len(reduct_pwd):][36:37]
            cat_fname  = spHect_flist[i][len(reduct_pwd):][7:35] + '.cat'
        elif year == '2025':
            config_str = spHect_flist[i][len(reduct_pwd):][7:14] + '-' + spHect_flist[i][len(reduct_pwd):][40:41]
            cat_fname  = spHect_flist[i][len(reduct_pwd):][7:39] + '.cat'
        # print(f"   -> configure str : {config_str}")
        # print(f"   -> cat file name : {cat_fname}")
        config_strs.append(config_str)
        cat_fnames.append(cat_fname)

    if len(spHect_flist) == 0:
        print(f" ! No spHect-NGC*.fits files found. Skipping {date}.")
        continue


    ## -------------------------------------------------- ##
    ## ----  1 - (2) Read spectra from the files  ------- ##
    ## -------------------------------------------------- ##

    print(f"\n --- Read spHect*.fits Files --- ")
    print(f" * observation date : {date}")
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

    fibTabPwd   = reduct_pwd
    fibTabFname = "spHect-NGC*.csv"
    fibTabFlist = sorted(glob.glob(reduct_pwd + fibTabFname))

    print(f"\n --- Reading spHect*_FibInfo_catID_added.csv Files --- ")
    print(f" * pwd             : {fibTabPwd}")

    fibTabList = []
    for i in range(len(fibTabFlist)):
        print(f"  - fibTabFlist[{i}] : {fibTabFlist[i][len(fibTabPwd):]}")
        fibTab = ascii.read(fibTabFlist[i])
        fibTabList.append(fibTab)


    ## -------------------------------------------------- ##
    ## ----  1 - (4) Calculate Gaussian-smoothed Spectra  ##
    ## -------------------------------------------------- ##

    gs_flux_lists = []

    for i in range(len(spectrum_lists)):
        spectrum_list = spectrum_lists[i]
        gs_flux_list  = []

        for j in range(len(spectrum_list)):
            flux = spectrum_list[j,1]
            wav  = spectrum_list[j,0]

            spec1d = Spectrum1D(flux=flux*u.Jy, spectral_axis=wav*u.Angstrom)
            spec1d = manipulation.gaussian_smooth(spectrum=spec1d, stddev=stddev)

            flux_gs = spec1d.flux.value

            gs_flux_list.append(flux_gs)

        gs_flux_list = np.array(gs_flux_list)
        gs_flux_lists.append(gs_flux_list)


    ## -------------------------------------------------- ##
    ## ----  2 - (2) Find known-specz galaxies  --------- ##
    ## -------------------------------------------------- ##

    print(f"\n --- Get Data of Already Known Spec-z Galaxies --- ")

    specz_gals_list = []

    for i in range(len(spectrum_lists)):
        cfg_str = config_strs[i]
        tName   = np.char.split(cfg_str, '-').tolist()[0]

        specz_gals = sptools.get_specz_known_gals(year, tName)

        fibIdxList = []
        ## --- Find corresponding Fiber Idxs --- ##
        for j in range(len(specz_gals)):
            specz_ID = specz_gals['ID'][j]
            try:
                fibIdx   = np.where(fibTabList[i]['catID'] == str(specz_ID))[0][0]
            except IndexError:
                print(f"   ! Spec-z galaxy ID {specz_ID} not found in fiber table.")
                print(f"    -> set the 'fibIdx' as None.")
                fibIdx = None
            fibIdxList.append(fibIdx)
        ## ------------------------------- ##

        specz_gals['fibIdx'] = fibIdxList
        specz_gals_list.append(specz_gals)

        # print(f" => specz_gals_list[{i}] >>")
        # print(specz_gals_list[i])


    ## -------------------------------------------------- ##
    ## ----  3 - (3) Measure redshifts  ----------------- ##
    ## -------------------------------------------------- ##

    df_results_list = []
    result_list = []
    for i in range(len(spectrum_lists)):
        cfg_str = config_strs[i]
        tName   = np.char.split(cfg_str, '-').tolist()[0]

        print(f" << cfg = {cfg_str} >> ")

        print(f"  {'DESI Legacy ID':^20}  {'SDSS redshift':^20}  {'# of measurements':^20}  {'best template':^20}  {'best redshift':^20}")
        print(f"  {'-'*20}  {'-'*20}  {'-'*20}  {'-'*20}  {'-'*20}")

        df_results = []

        for j in range(len(specz_gals_list[i])):
            galID   = specz_gals_list[i]['ID'].iloc[j]
            fib_idx = specz_gals_list[i]['fibIdx'][j]

            ## --- Measure --- ##
            measure = rvm(all_temps, n_jobs=n_jobs)
            if not np.isnan(fib_idx):
                fib_idx = int(fib_idx)
                df      = measure.z_single(spectrum_lists[i][fib_idx], mask=mask, 
                                           chi_thres=chi_thres, r_thres=r_thres, spectrum_range=spectrum_range)
            else:
                df = pd.DataFrame()  # Empty DataFrame if fib_idx is None
            ## --------------- ##

            sdss_z        = specz_gals_list[i]['s3_z'][j]
            n_measure     = len(df)
            tmpResultList = []

            if n_measure >= 1:
                best_template = df['template_name'][df['best'] == 'best'].values[0]
                best_z        = df['z'][df['best'] == 'best'].values[0]

                ## --- for diagnostic plot --- ##
                for tName in df['template_name']:
                    tmpResult = measure.cc_result[tName]
                    tmpResultList.append(tmpResult)
            else:
                best_template = 'N/A'
                best_z        = 'N/A'

            df_results.append(df)
            result_list.append(tmpResultList)

            print(f"  {galID:<20}" +
                  f"  {sdss_z:<20.5f}" +
                  f"  {n_measure:<20}" +
                  f"  {best_template:<20}" +
                  f"  {best_z:<20}")

        df_results_list.append(df_results)
        print(f"")


    ## -------------------------------------------------- ##
    ## ----  3 - (6) Save spectra plots  ---------------- ##
    ## -------------------------------------------------- ##

    print(f"\n --- Save Spectra Plots --- ")

    for cfg_idx in range(len(spectrum_lists)):

        print(f" << cfg = {config_strs[cfg_idx]} >> ")

        for specGal_idx in range(len(specz_gals_list[cfg_idx])):
            fib_idx = specz_gals_list[cfg_idx]['fibIdx'][specGal_idx]
            sdss_z  = specz_gals_list[cfg_idx]['s3_z'][specGal_idx]

            # print(f"   * DESI ID : {specz_gals_list[cfg_idx]['ID'][specGal_idx]} ")

            try:
                best_z = df_results_list[cfg_idx][specGal_idx]['z'][df_results_list[cfg_idx][specGal_idx]['best'] == 'best'].values[0]
            except:
                best_z = None

            sav_fname = f"{date}_zResult_{config_strs[cfg_idx]}_fibID_{fib_idx}_catID_{specz_gals_list[cfg_idx]['ID'][specGal_idx]}.png"

            ## --- Plot --- ##
            if not np.isnan(fib_idx):
                fib_idx = int(fib_idx)
                fig, ax = sptools.plot_spectrum(spectrum_lists[cfg_idx][fib_idx], gs_flux=gs_flux_lists[cfg_idx][fib_idx], masks=mask)
                fig, ax = sptools.add_lines(fig, ax, best_z, sdss_z, telluric_lines)

                if best_z is None: bestz_str = f"best z = N/A \n(failed to measure)"
                else:              bestz_str = f"best z = {best_z:.6f}"

                ax.text(0.95, 0.87, f"SDSS z = {sdss_z:.6f}", c='k', va='top', ha='right', transform=ax.transAxes, fontweight='normal')
                ax.text(0.95, 0.80, bestz_str, c='k', va='top', ha='right', transform=ax.transAxes, fontweight='normal')

                fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.90)
                ## ------------ ##

                ## --- Save Figure --- ##
                fExists = os.path.exists(sav_pwd_figs + sav_fname)
                if fExists:
                    print(f" * DESI ID: {specz_gals_list[cfg_idx]['ID'][specGal_idx]} -> Fig already exists. Not Saving ... ")
                else:
                    fig.savefig(sav_pwd_figs + sav_fname, dpi=300)
                    print(f" * DESI ID: {specz_gals_list[cfg_idx]['ID'][specGal_idx]} -> Saved!")
                ## ------------------- ##

                plt.close(fig)
            else:
                print(f"    -> fib_idx is NaN. Skipping ... ")


    ## -------------------------------------------------- ##
    ## ----  4 - (1) Collect redshift measurements  ----- ##
    ## -------------------------------------------------- ##

    SDSS_zs = []
    best_zs = []
    for i in range(len(specz_gals_list)):
        for j in range(len(specz_gals_list[i])):
            sdss_z = specz_gals_list[i]['s3_z'][j]
            try:
                best_z = df_results_list[i][j]['z'][df_results_list[i][j]['best'] == 'best'].values[0]
            except:
                best_z = None

            if best_z is not None:
                SDSS_zs.append(sdss_z)
                best_zs.append(best_z)

    all_SDSS_zs.extend(SDSS_zs)
    all_best_zs.extend(best_zs)


    ## -------------------------------------------------- ##
    ## ----  4 - (2) Per-date comparison plot  ---------- ##
    ## -------------------------------------------------- ##

    # if len(SDSS_zs) > 0:
    #     fig, ax = plt.subplots(figsize=(5,5))

    #     ax.scatter(SDSS_zs, best_zs, marker='x', c='r', s=20, alpha=0.7)
    #     ax.plot([0, 1], [0, 1], c='k', ls='--', lw=1)

    #     ax.set_xlabel(f"Redshift (SDSS)")
    #     ax.set_ylabel(f"Redshift (RVSNUpy)")

    #     ax.set_xlim(0, 0.2)
    #     ax.set_ylim(0, 0.2)

    #     fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)

    #     sav_fname = f"{date}_zResults_comparison_SDSS_vs_RVSNUpy.png"
    #     fExists   = os.path.exists(sav_pwd_figs + sav_fname)
    #     if fExists:
    #         print(f" -> Fig already exists. Not Saving ... ")
    #     else:
    #         fig.savefig(sav_pwd_figs + sav_fname, dpi=300)
    #         print(f" -> Saved! ({sav_fname})")
    #     plt.close(fig)


    ## -------------------------------------------------- ##
    ## ----  4 - (3) Per-date dz histogram  ------------- ##
    ## -------------------------------------------------- ##

    # if len(SDSS_zs) > 0:
    #     fig, ax = plt.subplots(figsize=(6,5))

    #     dz_bins = np.linspace(-0.0004, 0.0004, 20)

    #     dzs = np.array(best_zs) - np.array(SDSS_zs)
    #     ax.hist(dzs, bins=dz_bins, color='k', histtype='step', lw=1.5)

    #     ## --- velocity axis --- ##
    #     ax2 = ax.twiny()
    #     c = 3.0e5  # speed of light in km/s
    #     dz_vel_bins = dz_bins * c
    #     ax2.set_xlabel(r"$\Delta v$ [km/s]")
    #     ax2.set_xlim(dz_bins[0]*c, dz_bins[-1]*c)
    #     ## --------------------- ##

    #     ax.text(0.07, 0.93, f"Ngal = {len(dzs)}", transform=ax.transAxes, va='top', ha='left', fontsize=15)

    #     ax.set_xlabel(r"$\Delta z$ (RVSNUpy - SDSS)", size=20)
    #     ax.set_ylabel("Counts", size=20)

    #     fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.85)

    #     sav_fname = f"{date}_zResults_dzHistogram_SDSS_vs_RVSNUpy.png"
    #     fExists   = os.path.exists(sav_pwd_figs + sav_fname)
    #     if fExists:
    #         print(f" -> Fig already exists. Not Saving ... ")
    #     else:
    #         fig.savefig(sav_pwd_figs + sav_fname, dpi=300)
    #         print(f" -> Saved! ({sav_fname})")
    #     plt.close(fig)


    ## -------------------------------------------------- ##
    ## ----  5 - (1) Make a table for known-specz gals  - ##
    ## -------------------------------------------------- ##

    specz_gals_withResults_list = []

    for i in range(len(specz_gals_list)):
        config_str = config_strs[i]

        nan_template_row = pd.DataFrame({
            'template_name': [np.nan],
            'z': [np.nan],
            'zerr': [np.nan],
            'r': [np.nan],
            'chi_eff': [np.nan],
            'best': [np.nan],
            'note': [np.nan]
        })

        rows = []
        for j in range(len(specz_gals_list[i])):
            gal_meta = specz_gals_list[i].iloc[j].to_dict()

            try:
                all_measures = df_results_list[i][j]
            except:
                all_measures = pd.DataFrame()

            if len(all_measures) == 0:
                all_measures = nan_template_row.copy()

            all_measures = all_measures.rename(columns={
                'template_name': 'template_RV',
                'z'            : 'z_RV',
                'zerr'         : 'zerr_RV',
                'r'            : 'r_RV',
                'chi_eff'      : 'chi_eff_RV',
                'best'         : 'best_RV',
                'note'         : 'note_RV',
            })

            for _, temp_row in all_measures.iterrows():
                rows.append({**gal_meta, **temp_row.to_dict()})

        specz_gals_withResults = pd.DataFrame(rows)
        specz_gals_withResults_list.append(specz_gals_withResults)

        # print(f"    -> specz_gals_withResults[{i}] ")
        # print(specz_gals_withResults)


    ## -------------------------------------------------- ##
    ## ----  5 - (2) Collect results with date & cfg  --- ##
    ## -------------------------------------------------- ##

    for i in range(len(specz_gals_withResults_list)):
        config_str = config_strs[i]
        df = specz_gals_withResults_list[i].copy()
        df.insert(0, 'obs_date', date)
        df.insert(1, 'cfg_str', config_str)
        all_specz_gals_withResults.append(df)


## ====================================================== ##
## ====  Overall comparison plots (all dates)  ========= ##
## ====================================================== ##

if len(all_SDSS_zs) > 0:

    ## --- overall SDSS vs RVSNUpy comparison --- ##
    fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(all_SDSS_zs, all_best_zs, marker='x', c='r', s=20, alpha=0.7)
    ax.plot([0, 1], [0, 1], c='k', ls='--', lw=1)

    ax.set_xlabel(f"Redshift (SDSS)")
    ax.set_ylabel(f"Redshift (RVSNUpy)")

    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)

    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)

    sav_fname = f"zResults_comparison_SDSS_vs_RVSNUpy.png"
    fExists   = os.path.exists(sav_pwd_figs + sav_fname)
    if fExists:
        print(f" -> Fig already exists. Not Saving ... ")
    else:
        fig.savefig(sav_pwd_figs + sav_fname, dpi=300)
        print(f" -> Saved! ({sav_fname})")
    plt.close(fig)


    ## --- overall dz histogram --- ##
    fig, ax = plt.subplots(figsize=(6,5))

    dz_bins = np.linspace(-0.0004, 0.0004, 20)

    dzs = np.array(all_best_zs) - np.array(all_SDSS_zs)
    ax.hist(dzs, bins=dz_bins, color='k', histtype='step', lw=1.5)

    ## --- velocity axis --- ##
    ax2 = ax.twiny()
    c = 3.0e5  # speed of light in km/s
    dz_vel_bins = dz_bins * c
    ax2.set_xlabel(r"$\Delta v$ [km/s]")
    ax2.set_xlim(dz_bins[0]*c, dz_bins[-1]*c)
    ## --------------------- ##

    ax.text(0.07, 0.93, f"Ngal = {len(dzs)}", transform=ax.transAxes, va='top', ha='left', fontsize=15)

    ax.set_xlabel(r"$\Delta z$ (RVSNUpy - SDSS)", size=20)
    ax.set_ylabel("Counts", size=20)

    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.85)

    sav_fname = f"zResults_dzHistogram_SDSS_vs_RVSNUpy.png"
    fExists   = os.path.exists(sav_pwd_figs + sav_fname)
    if fExists:
        print(f" -> Fig already exists. Not Saving ... ")
    else:
        fig.savefig(sav_pwd_figs + sav_fname, dpi=300)
        print(f" -> Saved! ({sav_fname})")
    plt.close(fig)


## ====================================================== ##
## ====  Save combined zResults_knownSpecZ.dat  ========= ##
## ====================================================== ##

print(f"\n = = = Save Combined Results = = = ")
print(f" * sav pwd : {sav_pwd_dat} ")

if len(all_specz_gals_withResults) > 0:
    combined_df  = pd.concat(all_specz_gals_withResults, ignore_index=True)
    combined_cat = Table.from_pandas(combined_df)

    sav_fname = f"zResults_knownSpecZ.dat"
    ascii.write(combined_cat, sav_pwd_dat + sav_fname, format='fixed_width_two_line', overwrite=True)
    print(f" -> Saved! ({sav_fname})")
    print(f"    * N_measurements = {len(combined_df)}")
else:
    print(f" ! No results to save.")

print(f"\n = = = Done! = = = ")
