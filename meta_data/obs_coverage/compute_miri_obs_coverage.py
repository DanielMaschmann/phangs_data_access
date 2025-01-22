"""
Script to develop how to check the observational coverage of a PHANGS target
"""

import numpy as np
import os
import pickle
from phangs_data_access import phot_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info
import matplotlib.pyplot as plt


target_list = phangs_info.phangs_treasury_jwst_galaxy_list

for target in target_list:

    # if os.path.isfile('data_output/%s_miri_obs_hull_dict.npy' % target):
    #     continue

    # now get miri bands
    phangs_phot = phot_access.PhotAccess(target_name=target)

    # get band list
    band_list = helper_func.ObsTools.get_miri_obs_band_list(target=target)

    print(target, ' bands, available: ', band_list)
    phangs_phot.load_phangs_bands(band_list=band_list)

    obs_hull_dict = {}
    for band in band_list:

        img_data = phangs_phot.miri_bands_data['%s_data_img' % band]
        img_wcs = phangs_phot.miri_bands_data['%s_wcs_img' % band]

        mask_covered_pixels = np.array(np.invert(img_data == 0), dtype=float)
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
        obs_hull_dict.update({band: hull_coord_dict})
        # plt.show()

    # save dictionary
    if not os.path.isdir('data_output'):
        os.makedirs('data_output')

    with open('data_output/%s_miri_obs_hull_dict.npy' % target, 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)

