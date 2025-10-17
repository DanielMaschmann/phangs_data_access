"""
Script to develop how to check the observational coverage of a PHANGS target
"""

import numpy as np
import os
import pickle
from phangs_data_access import phot_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info
from phangs_data_access import phangs_access_config
import matplotlib.pyplot as plt


plot_flag = True

nircam_version = 'v1p1p1'

target_list = list(getattr(phangs_info, 'jwst_obs_band_dict_%s' % nircam_version).keys())

print(target_list)
for target in target_list:

    # if os.path.isfile('data_output/%s_nircam_obs_hull_dict.npy' % target):
    #     continue

    # now get nircam bands
    phangs_phot = phot_access.PhotAccess(phot_target_name=target, phot_nircam_target_name=target, nircam_data_ver=nircam_version)

    # get band list
    band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=target, version=nircam_version)

    print(target, ' bands, available: ', band_list)
    phangs_phot.load_phangs_bands(band_list=band_list)



    obs_hull_dict = {}
    for band in band_list:

        img_data = phangs_phot.nircam_bands_data['%s_data_img' % band]
        img_wcs = phangs_phot.nircam_bands_data['%s_wcs_img' % band]
        # plt.imshow(img_data)
        if nircam_version == 'v1p1p1':
            mask_covered_pixels = np.array(np.invert(img_data == 0), dtype=float)
        elif nircam_version == 'v2p0':
            mask_covered_pixels = np.array(np.invert(np.isnan(img_data)), dtype=float)
        elif nircam_version == 'v0p3':
            mask_covered_pixels = np.array(np.invert(np.isnan(img_data)), dtype=float)

        hull_dict = helper_func.GeometryTools.contour2hull(data_array=mask_covered_pixels,
                                                       level=0, contour_index=0, n_max_rejection_vertice=100)

        print(band, ' n of hulls: ', len(hull_dict.keys()))
        # some targets have saturated observations in the middle which we ignore
        if (target == 'ngc6300') & (band == 'F164N'):
            hull_dict = {0: hull_dict[0]}
        if (target == 'ngc1068') & (band in ['F300M', 'F335M']):
            hull_dict = {0: hull_dict[0]}
        # now save the hull points as coordinates
        hull_coord_dict = {}
        for idx in hull_dict.keys():
            # transform into coordinates
            coordinates = img_wcs.pixel_to_world(hull_dict[idx]['x_convex_hull'], hull_dict[idx]['y_convex_hull'])
            ra = coordinates.ra.deg
            dec = coordinates.dec.deg
            hull_coord_dict.update({idx: {'ra': ra, 'dec': dec}})
            # plt.plot(hull_dict[idx]['x_convex_hull'] - 1, hull_dict[idx]['y_convex_hull'] - 1)

        obs_hull_dict.update({band: hull_coord_dict})

        # plt.show()

    # save dictionary
    if not os.path.isdir('data_output'):
        os.makedirs('data_output')

    with open('data_output/%s_nircam_obs_hull_dict_%s.npy' % (target, nircam_version), 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)



