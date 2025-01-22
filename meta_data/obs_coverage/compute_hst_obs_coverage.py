"""
Script to develop how to check the observational coverage of a PHANGS target
"""

import numpy as np
import os
import pickle
from phangs_data_access import phot_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info


target_list = phangs_info.phangs_treasury_hst_galaxy_list


for target in target_list:

    # if os.path.isfile('data_output/%s_hst_obs_hull_dict.npy' % target):
    #     continue
    if target == 'ngc1510':
        continue

    # now get hst bands
    phangs_phot = phot_access.PhotAccess(phot_target_name=target)

    # get band list
    band_list = helper_func.ObsTools.get_hst_obs_band_list(target=target)

    print(target, ' bands, available: ', band_list)

    obs_hull_dict = {}
    for band in band_list:

        exp_file_name = phangs_phot.get_hst_img_file_name(band=band, file_type='wht')
        data, header, wcs = helper_func.FileTools.load_img(file_name=exp_file_name)
        mask_covered_pixels = data > 0

        hull_dict = helper_func.GeometryTools.contour2hull(data_array=data, level=0, contour_index=0, n_max_rejection_vertice=1000)

        print(band, ' n of hulls: ', len(hull_dict.keys()))

        # now save the hull points as coordinates
        hull_coord_dict = {}
        for idx in hull_dict.keys():
            # transform into coordinates
            coordinates = wcs.pixel_to_world(hull_dict[idx]['x_convex_hull'], hull_dict[idx]['y_convex_hull'])
            ra = coordinates.ra.deg
            dec = coordinates.dec.deg
            hull_coord_dict.update({idx: {'ra': ra, 'dec': dec}})
        obs_hull_dict.update({band: hull_coord_dict})

    # save dictionary
    if not os.path.isdir('data_output'):
        os.makedirs('data_output')

    with open('data_output/%s_hst_obs_hull_dict.npy' % target, 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)



