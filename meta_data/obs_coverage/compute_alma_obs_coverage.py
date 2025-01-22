"""
Script to develop how to check the observational coverage of a PHANGS target
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from phangs_data_access import gas_access
from phangs_data_access import helper_func
from phangs_data_access import phangs_info


target_list = phangs_info.phangs_alma_galaxy_list

obs_hull_dict = {}
for target in target_list:

    # if os.path.isfile('data_output/%s_alma_obs_hull_dict.npy' % target):
    #     continue

    print(target)
    # target = 'ngc1097'

    phangs_gas = gas_access.GasAccess(gas_target_name=target)

    for res in ['native', 150]:

        # check if data is available
        file_name = phangs_gas.get_alma_co21_mom_map_file_name(mom='mom0', res=res)
        alma_data_available = os.path.isfile(file_name)

        print(res, alma_data_available)

        if not alma_data_available:
            continue

        # get alma_data
        phangs_gas.load_alma_co21_data(mom='mom0', res=res)

        data = phangs_gas.alma_data['mom0_native_data_img']
        wcs = phangs_gas.alma_data['mom0_native_wcs_img']

        # the problem here is that the pixels with the data is directly bordering to the end of the image
        # and thus a hull is hard to compute from this.
        new_nan_data = np.zeros((data.shape[0] + 2, data.shape[1] + 2)) * np.nan
        new_nan_data[1:-1, 1:-1] = data
        mask_covered_pixels = np.invert(np.isnan(new_nan_data))

        hull_dict = helper_func.GeometryTools.contour2hull(data_array=mask_covered_pixels, level=0, contour_index=0, n_max_rejection_vertice=100)

        # plt.imshow(data)

        # now save the hull points as coordinates
        hull_coord_dict = {}
        for idx in hull_dict.keys():
            # plt.plot(hull_dict[idx]['x_convex_hull'] - 1, hull_dict[idx]['y_convex_hull'] - 1)
            # transform into coordinates
            # subtract 1 from x and y because of the resampling above
            coordinates = wcs.pixel_to_world(hull_dict[idx]['x_convex_hull'] - 1, hull_dict[idx]['y_convex_hull'] - 1)
            ra = coordinates.ra.deg
            dec = coordinates.dec.deg
            hull_coord_dict.update({idx: {'ra': ra, 'dec': dec}})
        obs_hull_dict.update({res: hull_coord_dict})

    # plt.show()

    # save dictionary
    if not os.path.isdir('data_output'):
        os.makedirs('data_output')

    with open('data_output/%s_alma_obs_hull_dict.npy' % target, 'wb') as file_name:
        pickle.dump(obs_hull_dict, file_name)

