"""
Construct a data access structure for all kind of X-ray observation
"""
import os.path
from pathlib import Path
import pickle

import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.constants import c as speed_of_light

from phangs_data_access import phangs_access_config, helper_func, phangs_info, phys_params
from phangs_data_access.sample_access import SampleAccess


class XAccess:
    """
    Access class to organize data structure of chnadra observations
    """

    def __init__(self, x_target_name=None):
        """

        Parameters
        ----------
        x_target_name : str
            Default None. Target name
        x_target_name : str
            Default None. Target name used for Hs observation
        """

        # get target specifications
        # check if the target names are compatible

        if x_target_name is not None:
            x_target_name = helper_func.FileTools.target_name_no_directions(target=x_target_name)

        # if (x_target_name not in phangs_info.phangs_alma_galaxy_list) & (x_target_name is not None):
        #     # raise AttributeError('The target %s is not in the PHANGS ALMA sample or has not been added to '
        #     #                      'the current package version' % x_target_name)

        self.x_target_name = x_target_name

        # loaded data dictionaries
        self.x_data = {}

        # get path to observation coverage hulls
        self.path2obs_cover_hull = (Path(__file__).parent.parent.absolute() / 'meta_data' / 'obs_coverage' /
                                    'data_output')

        super().__init__()

    def get_chandra_file_name(self, energy='0p5to2'):
        """

        Parameters
        ----------
        energy : str
        Returns
        -------
        data_file_path : ``Path``
        """
        assert energy in ['0p5to2', '0p5to7', '2to7']

        file_path = Path(phangs_access_config.phangs_config_dict['chandra_data_path'])
        if self.x_target_name == 'ngc5194':
            file_name = '%s-%s-asca-merged-im-bin1.fits' % (self.x_target_name.upper(), energy)
        else:
            file_name = '%s-%s-merged-img-bin1_astro.fits' % (self.x_target_name.upper(), energy)

        return file_path / file_name

    def load_chandra_data(self, energy='0p5to2', reload=False):
        """

        Parameters
        ----------
        energy : str
        reload : bool
        """
        # check if data is already loaded
        if (('%s_data_img' % energy) in self.x_data.keys()) & (not reload):
            return None

        file_name = self.get_chandra_file_name(energy=energy)

        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

        self.x_data.update({'%s_data_img' % energy: img_data,
                            '%s_header_img' % energy: img_header,
                            '%s_wcs_img' % energy: img_wcs,
                            '%s_unit_img' % energy: 'counts',
                            '%s_pixel_area_size_sr_img' % energy:
                                img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
