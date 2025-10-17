"""
Function to
"""
from cluster_cat_dr.helper_func import download_file
import numpy as np
import os
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# UVIS1 is estimated for observations till MJD 55008 - June 26, 2009).
url_uvis1 = 'https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/data-analysis/' \
            'photometric-calibration/uvis-encircled-energy/_documents/wfc3uvis1_aper_007_syn.csv'
# after June 26, 2009, UVIS2 is used
url_uvis2 = 'https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/data-analysis/' \
            'photometric-calibration/uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv'

file_path_uvis1 = 'data/wfc3uvis1_aper_007_syn.csv'
file_path_uvis2 = 'data/wfc3uvis2_aper_007_syn.csv'
# make sure that aperture files are present
if not os.path.isdir('data'):
    os.makedirs('data')
download_file(file_path=file_path_uvis1, url=url_uvis1)
download_file(file_path=file_path_uvis2, url=url_uvis2)

# we exlude F218W and F225W as they have a very small PSF and there it is not really possible to extract the ee50% for
# these filters
hst_filter_list = ['F275W', 'F300X', 'F280N', 'F336W', 'F343N', 'F373N', 'F390M', 'F390W', 'F395N',
                   'F410M', 'F438W', 'F467M', 'F469N', 'F475W', 'F487N', 'F475X', 'F200LP', 'F502N', 'F555W', 'F547M',
                   'F350LP', 'F606W', 'F621M', 'F625W', 'F631N', 'F645N', 'F656N', 'F657N', 'F658N', 'F665N', 'F673N',
                   'F689M', 'F680N', 'F600LP', 'F763M', 'F775W', 'F814W', 'F845M', 'F850LP', 'F953N']


uvis1_table = ascii.read(file_path_uvis1)
uvis2_table = ascii.read(file_path_uvis2)


# get aperture values from colum names
col_names = uvis1_table.keys()
aperture_names = []
aperture_values = []
for name in col_names:
    if 'APER#' in name:
        aperture_names.append(name)
        aperture_values.append(float(name[5:]))
aperture_names = aperture_names
aperture_values = np.array(aperture_values)

uvis1_ee_dic = {}
uvis2_ee_dic = {}

if not os.path.isdir('plot_output'):
    os.makedirs('plot_output')

for filter in hst_filter_list:
    position_filter = np.where(uvis1_table['FILTER'] == filter)

    ee_value_uvis1 = np.array(list(uvis1_table[aperture_names][position_filter[0]][0]))
    ee_value_uvis2 = np.array(list(uvis2_table[aperture_names][position_filter[0]][0]))
    # we use a linear interpolation as we are only interested in the values around 50%
    interpolation_uvis1 = interp1d(ee_value_uvis1, aperture_values)
    interpolation_uvis2 = interp1d(ee_value_uvis2, aperture_values)
    ynew_uvis1 = np.linspace(min(ee_value_uvis1), max(ee_value_uvis1), 55)
    ynew_uvis2 = np.linspace(min(ee_value_uvis2), max(ee_value_uvis2), 55)

    ee_50_uvis1 = interpolation_uvis1(0.5)
    ee_80_uvis1 = interpolation_uvis1(0.8)
    uvis1_ee_dic.update({'%s' % filter: {'ee50': float(ee_50_uvis1), 'ee_80': float(ee_80_uvis1)}})
    ee_50_uvis2 = interpolation_uvis2(0.5)
    ee_80_uvis2 = interpolation_uvis2(0.8)
    uvis2_ee_dic.update({'%s' % filter: {'ee50': float(ee_50_uvis2), 'ee_80': float(ee_80_uvis2)}})

    print(filter, ' UVIS1 interp_fn_p50 ', float(ee_50_uvis1), ' interp_fn_p80 ', float(ee_80_uvis1))
    print(filter, ' UVIS2 interp_fn_p50 ', float(ee_50_uvis2), ' interp_fn_p80 ', float(ee_80_uvis2))

    # visualize aperture estimation

    fig, ax = plt.subplots(figsize=(9, 7))
    fontsize = 17

    ax.plot(interpolation_uvis1(ynew_uvis1), ynew_uvis1)
    ax.scatter(aperture_values, ee_value_uvis1)

    ax.plot(interpolation_uvis2(ynew_uvis2), ynew_uvis2)
    ax.scatter(aperture_values, ee_value_uvis2)

    ax.plot([0, ee_50_uvis1], [0.5, 0.5], color='k', linestyle='--')
    ax.plot([ee_50_uvis1, ee_50_uvis1], [0.0, 0.5], color='k', linestyle='--')
    ax.plot([0, ee_80_uvis1], [0.8, 0.8], color='k', linestyle='--')
    ax.plot([ee_80_uvis1, ee_80_uvis1], [0.0, 0.8], color='k', linestyle='--')
    ax.scatter(ee_50_uvis1, 0.5, color='k')
    ax.scatter(ee_80_uvis1, 0.8, color='k')


    ax.plot([0, ee_50_uvis2], [0.5, 0.5], color='k', linestyle='--')
    ax.plot([ee_50_uvis2, ee_50_uvis2], [0.0, 0.5], color='k', linestyle='--')
    ax.plot([0, ee_80_uvis2], [0.8, 0.8], color='k', linestyle='--')
    ax.plot([ee_80_uvis2, ee_80_uvis2], [0.0, 0.8], color='k', linestyle='--')
    ax.scatter(ee_50_uvis2, 0.5, color='k')
    ax.scatter(ee_80_uvis2, 0.8, color='k')

    ax.text(0.05, 0.83, filter, fontsize=fontsize)

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel('APP size [arcsec]', fontsize=fontsize)
    ax.set_ylabel('Fractionof EE', fontsize=fontsize)
    ax.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=fontsize-3)

    plt.savefig('plot_output/uvis1_2_ee_estimation_%s.png' % filter)
    plt.clf()
    plt.close()


print(uvis1_ee_dic)
print(uvis2_ee_dic)


# now we do the same thing for the ACS / WFC1
# description see https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections
# and Bohlin 2016AJ....152...60B
acs_wfc1_url = 'https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/data-analysis/' \
               'aperture-corrections/_documents/bohlin2016_wfc_ee-1.txt'

file_path_acs_wfc1_url = 'data/bohlin2016_wfc_ee-1.txt'
download_file(file_path=file_path_acs_wfc1_url, url=acs_wfc1_url)

data_wfc1 = np.genfromtxt(file_path_acs_wfc1_url, skip_header=4, dtype=object)

print(data_wfc1)

filter_names = np.array(data_wfc1[1:, 0], dtype=str)
aperture_sizes_pixel = np.array(data_wfc1[0, 1:], dtype=float)
# The active image area of each WFC detector is 4096 × 2048 pixel^2. The mean plate scale is 0.05 arcseconds/pixel,
# see https://hst-docs.stsci.edu/acsihb/chapter-7-observing-techniques/7-7-acs-apertures
pixel_scale = 0.05
aperture_sizes = aperture_sizes_pixel * pixel_scale
print(filter_names)
print(aperture_sizes)

wfc1_ee_dic = {}

for filter, filter_index in zip(filter_names, range(len(filter_names))):

    ee_value_wfc1 = np.array(data_wfc1[filter_index + 1, 1:], dtype=float)
    # we use a linear interpolation as we are only interested in the values around 50%
    interpolation_wfc1 = interp1d(ee_value_wfc1, aperture_sizes)
    ynew_wfc1 = np.linspace(min(ee_value_wfc1), max(ee_value_wfc1), 55)

    ee_50_wfc1 = interpolation_wfc1(0.5)
    ee_80_wfc1 = interpolation_wfc1(0.8)
    wfc1_ee_dic.update({'%s' % filter: {'ee50': float(ee_50_wfc1), 'ee_80': float(ee_80_wfc1)}})

    print(filter, ' UVIS1 interp_fn_p50 ', float(ee_50_wfc1), ' interp_fn_p80 ', float(ee_80_wfc1))

    # visualize aperture estimation

    fig, ax = plt.subplots(figsize=(9, 7))
    fontsize = 17

    ax.plot(interpolation_wfc1(ynew_wfc1), ynew_wfc1)
    ax.scatter(aperture_sizes, ee_value_wfc1)


    ax.plot([0, ee_50_wfc1], [0.5, 0.5], color='k', linestyle='--')
    ax.plot([ee_50_wfc1, ee_50_wfc1], [0.0, 0.5], color='k', linestyle='--')
    ax.plot([0, ee_80_wfc1], [0.8, 0.8], color='k', linestyle='--')
    ax.plot([ee_80_wfc1, ee_80_wfc1], [0.0, 0.8], color='k', linestyle='--')
    ax.scatter(ee_50_wfc1, 0.5, color='k')
    ax.scatter(ee_80_wfc1, 0.8, color='k')

    ax.text(0.05, 0.83, filter, fontsize=fontsize)

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel('APP size [arcsec]', fontsize=fontsize)
    ax.set_ylabel('Fractionof EE', fontsize=fontsize)
    ax.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=fontsize-3)

    plt.savefig('plot_output/wfc1_ee_estimation_%s.png' % filter)
    plt.clf()
    plt.close()

print(wfc1_ee_dic)

