"""
create all needed data for the HST PSFs
"""
import numpy as np
import os
from astropy.io import fits
from phangs_data_access import phot_tools, phot_access, helper_func
import matplotlib.pyplot as plt
import pickle
from astropy.visualization import SqrtStretch, SinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats



# parameters needed for procedure
n_scale_maps = 4
flux_unit = 'mJy'


target = 'ngc0628c'

# initialize data access
phangs_phot = phot_access.PhotAccess(phot_target_name=target)

hst_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=phangs_phot.phot_hst_target_name)
nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=phangs_phot.phot_nircam_target_name)
miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=phangs_phot.phot_miri_target_name)


print(hst_band_list)
print(nircam_band_list)
print(miri_band_list)

for hst_band in hst_band_list:
    print(hst_band)
    # load data
    phangs_phot.load_phangs_bands(band_list=[hst_band], flux_unit=flux_unit)
    img = phangs_phot.hst_bands_data['%s_data_img' % hst_band]
    wcs = phangs_phot.hst_bands_data['%s_wcs_img' % hst_band]

    # compute scale maps
    scale_map_list, residual_img, kernel_sizes_list = phot_tools.ScaleTools.constrained_diffusion_decomposition(
        img, max_n=n_scale_maps, verbosity=True)
    # create scale dict to
    scale_dict = {
        'n_scale_maps': n_scale_maps,
        'scale_map_list': scale_map_list,
        'residual_img': residual_img,
        'kernel_sizes_list': kernel_sizes_list,
        'flux_unit': flux_unit
    }

    # save scale maps
    with open('data_output/scale_map_dict_%s_n_scale_%i_flux_unit_%s.pickle' % (target, n_scale_maps, flux_unit.replace('/', '_')), 'wb') as file_name:
        pickle.dump(scale_dict, file_name)


    # plot scale maps
    fontsize = 23
    fig_size_individual = (5, 5)
    n_cols = 4
    n_rows = int(np.rint((n_scale_maps) / n_cols) + 1)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
                            figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))

    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    vmin = median - 1 * std
    vmax = median + 30 * std
    print(vmin, vmax)

    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)

    axs[0, 0].imshow(img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 0].set_title('original image', fontsize=fontsize)
    axs[0, 1].imshow(residual_img, origin='lower', norm=norm_img, cmap='Greys')
    axs[0, 1].set_title('Residuals', fontsize=fontsize)

    for idx_col in range(n_cols - 2):
        axs[0, idx_col + 2].axis('off')

    running_scale_idx = 0
    for idx_row in range(n_rows - 1):
        for idx_col in range(n_cols):
            axs[idx_row + 1, idx_col].imshow(scale_map_list[running_scale_idx], origin='lower', norm=norm_img,
                                             cmap='Greys')
            axs[idx_row + 1, idx_col].set_title('Scale %i' % (running_scale_idx + 1), fontsize=fontsize)
            running_scale_idx += 1

    fig.tight_layout()

    fig.savefig('plot_output/scale_decomposition_%s_%s_n_scale_%i.png' % (target, hst_band, n_scale_maps))
    plt.close(fig)
    plt.cla()

exit()






