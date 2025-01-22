"""
Construct a data access structure for CO and HI gas observations
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


class GasAccess:
    """
    Access class to organize data structure of ALMA observations
    """

    def __init__(self, gas_target_name=None):
        """

        Parameters
        ----------
        gas_target_name : str
            Default None. Target name
        gas_target_name : str
            Default None. Target name used for Hs observation
        """

        # get target specifications
        # check if the target names are compatible

        if gas_target_name is not None:
            gas_target_name = helper_func.FileTools.target_name_no_directions(target=gas_target_name)

        if (gas_target_name not in phangs_info.phangs_alma_galaxy_list) & (gas_target_name is not None):
            raise AttributeError('The target %s is not in the PHANGS ALMA sample or has not been added to '
                                 'the current package version' % gas_target_name)

        self.gas_target_name = gas_target_name

        # loaded data dictionaries
        self.alma_data = {}

        # get path to observation coverage hulls
        self.path2obs_cover_hull = (Path(__file__).parent.parent.absolute() / 'meta_data' / 'obs_coverage' /
                                    'data_output')

        super().__init__()

    def get_alma_co21_mom_map_file_name(self, mom='mom0', res=150):
        """

        Parameters
        ----------
        mom : str
        res : int or str
        Returns
        -------
        data_file_path : ``Path``
        """
        if isinstance(res, int):
            res_str = str(res) + 'pc_'
        elif res == 'native':
            res_str = ''
        else:
            raise KeyError('res must be either int with the resolution in pc or native')

        file_path = (Path(phangs_access_config.phangs_config_dict['alma_data_path']) /
                     ('delivery_%s' % phangs_access_config.phangs_config_dict['alma_data_ver']) / self.gas_target_name)
        file_name = '%s_12m+7m+tp_co21_%sbroad_%s.fits' % (self.gas_target_name, res_str, mom)

        return file_path / file_name

    def get_alma_co21_conv_map_file_name(self, alpha_co_method='S20_MUSEGPR'):
        """

        Parameters
        ----------
        alpha_co_method : str
            can be S20_MUSEGPR, S20_scaling, B13_MUSEGPR, B13_scaling,
            see Sun+ (2020, ApJ, 892, 148) or the read me file
        Returns
        -------
        data_file_path : ``Path``
        """

        file_path = (Path(phangs_access_config.phangs_config_dict['alma_conv_map_data_path']) /
                     phangs_access_config.phangs_config_dict['alma_conv_map_data_ver'])
        file_name = '%s_alphaCO21_%s.fits' % (self.gas_target_name.upper(), alpha_co_method)

        return file_path / file_name

    def load_alma_co21_data(self, mom='mom0', res=150, reload=False):
        """

        Parameters
        ----------
        mom : str
        res : int or str
        reload : bool
        """
        # check if data is already loaded
        if (('%s_%s_data_img' % (mom, str(res))) in self.alma_data.keys()) & (not reload):
            return None

        file_name = self.get_alma_co21_mom_map_file_name(mom=mom, res=res)

        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

        self.alma_data.update({'%s_%s_data_img' % (mom, str(res)): img_data,
                               '%s_%s_header_img' % (mom, str(res)): img_header,
                               '%s_%s_wcs_img' % (mom, str(res)): img_wcs,
                               '%s_%s_unit_img' % (mom, str(res)): 'K*km/s',
                               '%s_%s_pixel_area_size_sr_img' % (mom, str(res)):
                                   img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_alma_conv_map(self, alpha_co_method='S20_MUSEGPR', reload=False):
        """
        PHANGS-ALMA CO-to-H2 Conversion Factor Maps

        Parameters
        ----------
        alpha_co_method : str
        reload : bool

        """
        if ('alpha_co21_data_img' in self.alma_data.keys()) & (not reload):
            return None
        file_name = self.get_alma_co21_conv_map_file_name(alpha_co_method=alpha_co_method)

        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

        self.alma_data.update({'alpha_co21_data_img': img_data, 'alpha_co21_header_img': img_header,
                               'alpha_co21_wcs_img': img_wcs, 'alpha_co21_unit_img': 'K*km/s',
                               'alpha_co21_pixel_area_size_sr_img':
                                   img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def get_alma_h2_map(self, res=150, alpha_co_method='S20_MUSEGPR'):
        """

        Parameters
        ----------
        res : int or str
        alpha_co_method : str

        """
        self.load_alma_co21_data(res=res)
        self.load_alma_conv_map(alpha_co_method=alpha_co_method)

        # reproject conversion map to alma co map
        conv_map_reprojected = helper_func.CoordTools.reproject_image(
            data=self.alma_data['alpha_co21_data_img'], wcs=self.alma_data['alpha_co21_wcs_img'],
            new_wcs=self.alma_data['mom0_%s_wcs_img' % str(res)],
            new_shape=self.alma_data['mom0_%s_data_img' % str(res)].shape)

        return (self.alma_data['mom0_%s_data_img' % str(res)] * conv_map_reprojected,
                self.alma_data['mom0_%s_wcs_img' % str(res)])

    def get_alma_co21_cloud_cat_file_name(self, res=150):
        """
        Parameters
        ----------
        res : int or str
        Returns
        -------
        data_file_path : ``Path``
        """
        # To do: add homogenized maps
        if isinstance(res, int):
            res_folder = 'matched_%spc' % str(res)
            res_str = '%spc_nativenoise' % str(res)
        elif res == 'native':
            res_folder = 'native'
            res_str = 'native'

        else:
            raise KeyError('res must be either int with the resolution in pc or native')

        '/media/benutzer/Extreme Pro/data/phangs_data_products/cloud_catalogs/v4p0_ST1p6/v4p0_gmccats/matched_150pc'
        file_path = (Path(phangs_access_config.phangs_config_dict['alma_cloud_cat_data_path']) /
                     ('%s_%s' % (phangs_access_config.phangs_config_dict['alma_data_ver'],
                                 phangs_access_config.phangs_config_dict['alma_cloud_cat_data_release_ver'])) /
                     ('%s_gmccats' % phangs_access_config.phangs_config_dict['alma_data_ver']) / res_folder)
        file_name = '%s_12m+7m+tp_co21_%s_props.fits' % (self.gas_target_name, res_str)

        return file_path / file_name

    def load_alma_cloud_cat(self, res=150, reload=False):
        """
        best description can be found in Rosolowsky+2021 2021MNRAS.502.1218R
        Parameters
        ----------
        res : int or str
        reload : bool

        """
        if ('alpha_cloud_cat_data' in self.alma_data.keys()) & (not reload):
            return None
        file_name = self.get_alma_co21_cloud_cat_file_name(res=res)
        cat_data, cat_header = helper_func.FileTools.load_fits_table(file_name=file_name, hdu_number=1)

        self.alma_data.update({'alpha_cloud_cat_data': cat_data, 'alpha_cloud_cat_header': cat_header})

    def get_cloud_coords(self, res=150):
        """
        Get positions from giant molecular clouds
        Parameters
        ----------
        res : int or str

        """
        # load cloud catalog
        self.load_alma_cloud_cat(res=res)
        return self.alma_data['alpha_cloud_cat_data']['XCTR_DEG'], self.alma_data['alpha_cloud_cat_data']['YCTR_DEG']

    def get_cloud_rad_pc(self, res=150):
        """
        Get radius from giant molecular clouds in pc
        Parameters
        ----------
        res : int or str

        """
        # load cloud catalog
        self.load_alma_cloud_cat(res=res)
        return self.alma_data['alpha_cloud_cat_data']['RAD3D_PC']

    def get_cloud_rad_arcsec(self, res=150):
        """
        Get radius from giant molecular clouds in arcsec
        Parameters
        ----------
        res : int or str

        """
        rad_cloud_pc = self. get_cloud_rad_pc(res=res)
        sample_access = SampleAccess()
        target_dist_mpc = sample_access.get_target_dist(target=self.gas_target_name)
        central_target_pos = helper_func.CoordTools.get_target_central_simbad_coords(target_name=self.gas_target_name,
                                                                                     target_dist_mpc=target_dist_mpc)

        off_set_central_pos = SkyCoord(ra=central_target_pos.ra + 1*u.arcsec, dec=central_target_pos.dec,
                                       distance=target_dist_mpc*u.Mpc)
        value_pc_per_arc_sec = central_target_pos.separation_3d(off_set_central_pos).to(u.pc).value
        return rad_cloud_pc / value_pc_per_arc_sec

    def get_cloud_surf_dens(self, res=150):
        """
        Get surface density from giant molecular clouds
        Parameters
        ----------
        res : int or str

        """
        # load cloud catalog
        self.load_alma_cloud_cat(res=res)
        return self.alma_data['alpha_cloud_cat_data']['SURFDENS']

    def get_alma_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of ALMA observations

        Returns
        -------
        coverage_dict : dict
        """
        # return np.load(self.path2obs_cover_hull / ('%s_alma_obs_hull_dict.npy' % self.gas_target_name),
        #                allow_pickle=True).item()
        with open(self.path2obs_cover_hull / ('%s_alma_obs_hull_dict.npy' % self.gas_target_name), 'rb') as file_name:
            return pickle.load(file_name)

    def check_coords_covered_by_alma(self, ra, dec, res='native', max_dist_dist2hull_arcsec=2):
        """
        Function to check if coordinate points are inside MUSE observation

        Parameters
        ----------
        ra : float or ``np.ndarray``
        dec : float or ``np.ndarray``
        res : str
        max_dist_dist2hull_arcsec : float

        Returns
        -------
        coverage_dict : ``np.ndarray``
        """

        if isinstance(ra, float):
            ra = [ra]
            dec = [dec]

        hull_dict = self.get_alma_obs_coverage_hull_dict()[res]
        coverage_mask = np.zeros(len(ra), dtype=bool)
        hull_data_ra = np.array([])
        hull_data_dec = np.array([])

        for hull_idx in hull_dict.keys():
            ra_hull = hull_dict[hull_idx]['ra']
            dec_hull = hull_dict[hull_idx]['dec']
            hull_data_ra = np.concatenate([hull_data_ra, ra_hull])
            hull_data_dec = np.concatenate([hull_data_dec, dec_hull])
            coverage_mask += helper_func.GeometryTools.check_points_in_polygon(x_point=ra, y_point=dec,
                                                                               x_data_hull=ra_hull,
                                                                               y_data_hull=dec_hull)

        coverage_mask *= helper_func.GeometryTools.flag_close_points2ensemble(
            x_data=ra, y_data=dec, x_data_ensemble=hull_data_ra,y_data_ensemble=hull_data_dec,
            max_dist2ensemble=max_dist_dist2hull_arcsec/3600)

        return coverage_mask
