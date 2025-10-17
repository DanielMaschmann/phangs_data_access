"""
Script to develop how to check the observational coverage of a PHANGS target
"""

import numpy as np
import os
import pickle
from phangs_data_access import phot_access, helper_func, phangs_info
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, SinhStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats


plot_flag = True

miri_version = 'v1p1p1'

target_list = list(getattr(phangs_info, 'jwst_obs_band_dict_%s' % miri_version).keys())

print(target_list)

for target in target_list:

    # if os.path.isfile('data_output/%s_miri_obs_hull_dict.npy' % target):
    #     continue

    # now get miri bands
    phangs_phot = phot_access.PhotAccess(phot_target_name=target, phot_miri_target_name=target, miri_data_ver=miri_version)

    # get band list
    band_list = helper_func.ObsTools.get_miri_obs_band_list(target=target, version=miri_version)

    print(target, ' bands, available: ', band_list)

    phangs_phot.load_phangs_bands(band_list=band_list)

    # for plotting...
    min_ra = 361
    max_ra = -1
    min_dec = 181
    max_dec = -181

    obs_hull_dict = {}

    for band in band_list:

        img_data = phangs_phot.miri_bands_data['%s_data_img' % band]
        img_wcs = phangs_phot.miri_bands_data['%s_wcs_img' % band]

        # plt.imshow(img_data)

        if miri_version == 'v1p1p1':
            mask_covered_pixels = np.array(np.invert(img_data == 0), dtype=float)
        elif miri_version == 'v2p0':
            mask_covered_pixels = np.array(np.invert(np.isnan(img_data)), dtype=float)
        elif miri_version == 'v0p2':
            mask_covered_pixels = np.array(np.invert(np.isnan(img_data)), dtype=float)



        hull_dict = helper_func.GeometryTools.contour2hull(data_array=mask_covered_pixels,
                                                           level=0, contour_index=0, n_max_rejection_vertice=1000)

        print(band, ' n of hulls: ', len(hull_dict.keys()))

        # now save the hull points as coordinates
        hull_coord_dict = {}
        for idx in hull_dict.keys():
            # transform into coordinates
            coordinates = img_wcs.pixel_to_world(hull_dict[idx]['x_convex_hull'], hull_dict[idx]['y_convex_hull'])
            ra = coordinates.ra.deg
            dec = coordinates.dec.deg
            # plt.plot(hull_dict[idx]['x_convex_hull'] - 1, hull_dict[idx]['y_convex_hull'] - 1)
            hull_coord_dict.update({idx: {'ra': ra, 'dec': dec}})

            min_ra = np.min((min_ra, np.min(ra)))
            max_ra = np.max((min_ra, np.max(ra)))
            min_dec = np.min((min_dec, np.min(dec)))
            max_dec = np.max((min_dec, np.max(dec)))

        obs_hull_dict.update({band: hull_coord_dict})
        # plt.show()


    # save dictionary
    if not os.path.isdir('data_output'): os.makedirs('data_output')

    with open('data_output/%s_miri_obs_hull_dict_%s.npy' % (target, miri_version), 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)

    # if plot_flag:
    #     if not os.path.isdir('plot_output'): os.makedirs('plot_output')
    #
    #     # get maximal size of observations
    #     max_size_deg = np.max((max_ra - min_ra, (max_dec - min_dec) / np.cos(np.mean(dec) * np.pi / 180)))
    #
    #     # print(max_size_deg)
    #     # exit()
    #     print(max_size_deg * 3600)
    #
    #     img_dss, wcs_dss = phangs_phot.get_dss_img(img_rad_arcsec=60, survey='DSS', pixels_size=(500, 500))
    #     fig = plt.figure(figsize=(20, 20))
    #     ax_img = fig.add_axes((0.05, 0.05, 0.94, 0.94), projection = wcs_dss)
    #     mean, median, std = sigma_clipped_stats(img_dss, sigma=3.0)
    #     vmin = median - 1 * std
    #     vmax = median + 30 * std
    #     print(vmin, vmax)
    #     norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
    #     ax_img.imshow(img_dss, norm=norm_img, cmap='Greys')
    #     fig.savefig('plot_output/obs_cover_%s_miri.png' % target)
    #     plt.close(fig)
    #     exit()

