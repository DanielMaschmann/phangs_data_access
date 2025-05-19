"""
Data access structure for all kind of spectroscopic data products related to PHANGS
"""

from pathlib import Path
import numpy as np
import pickle

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from spectral_cube import SpectralCube

from TardisPipeline.readData.MUSE_WFM import get_MUSE_polyFWHM
from phangs_data_access import phangs_access_config, helper_func, phangs_info


class SpecAccess:
    """
    Access class to organize all kind of spectroscopic data available
    """

    def __init__(self, spec_target_name=None):
        """

        Parameters
        ----------
        spec_target_name : str
            Default None. Target name
        """
        if spec_target_name is not None:
            spec_target_name = helper_func.FileTools.target_name_no_directions(target=spec_target_name)

        self.spec_target_name = spec_target_name

        # loaded data dictionaries
        self.muse_dap_map_data = {}
        self.muse_datacube_data = {}
        self.kcwi_datacube_data = {}

        # get path to observation coverage hulls
        self.path2obs_cover_hull = (Path(__file__).parent.parent.absolute() / 'meta_data' / 'obs_coverage' /
                                    'data_output')

        super().__init__()

    def get_muse_data_file_name(self, data_prod='MUSEDAP', res='copt', ssp_model='fiducial'):
        """

        Parameters
        ----------
        data_prod : str
        res : str
        ssp_model : str
            This one is only for the copt DAPMAP data products

        Return
        ---------
        fiel_path : str
        """

        # get folder path
        file_path = (Path(phangs_access_config.phangs_config_dict['muse_data_path']) /
                     phangs_access_config.phangs_config_dict['muse_data_ver'] / res / data_prod)
        if (res == 'copt') & (data_prod == 'MUSEDAP'):
            file_path /= ssp_model

        # get file name
        if data_prod == 'MUSEDAP':
            if res == 'copt':
                file_name = '%s-%.2fasec_MAPS.fits' % (self.spec_target_name.upper(),
                                                       phangs_info.muse_obs_res_dict[self.spec_target_name]['copt_res'])
            elif res == 'native':
                file_name = '%s_MAPS.fits' % self.spec_target_name.upper()
            elif res == '150pc':
                file_name = '%s-150pc_MAPS.fits' % self.spec_target_name.upper()
            elif res == '15asec':
                file_name = '%s-15asec_MAPS.fits' % self.spec_target_name.upper()
            else:
                raise KeyError(res, ' must be copt, native, 150pc or 15asec')
        elif data_prod == 'datacubes':
            if res == 'copt':
                file_name = '%s-%.2fasec.fits' % (self.spec_target_name.upper(),
                                                       phangs_info.muse_obs_res_dict[self.spec_target_name]['copt_res'])
            elif res == 'native':
                file_name = '%s.fits' % self.spec_target_name.upper()
            elif res == '150pc':
                file_name = '%s-150pc.fits' % self.spec_target_name.upper()
            elif res == '15asec':
                file_name = '%s-15asec.fits' % self.spec_target_name.upper()
            else:
                raise KeyError(res, ' must be copt, native, 150pc or 15asec')
        else:
            raise KeyError(data_prod, ' must be either DAPMAP or datacubes')

        return file_path / file_name

    def get_kcwi_data_file_name(self):
        """

        Parameters
        ----------
        data_prod : str
        res : str
        ssp_model : str
            This one is only for the copt DAPMAP data products

        Return
        ---------
        fiel_path : str
        """

        # get folder path
        file_path = (Path(phangs_access_config.phangs_config_dict['kcwi_data_path']) / 'Datacubes')

        file_name = helper_func.FileTools.target_names_no_zeros(target=self.spec_target_name).upper() + '.fits'

        return file_path / file_name

    def load_muse_dap_map(self, res='copt', ssp_model='fiducial', map_type='HA6562_FLUX'):
        """

        Parameters
        ----------
        res : str
        ssp_model : str
            This one is only for the copt DAP results
        map_type : str

        """
        file_path = self.get_muse_data_file_name(res=res, ssp_model=ssp_model)

        # get MUSE data
        muse_hdu = fits.open(file_path)
        muse_map_data = muse_hdu[map_type].data
        muse_map_wcs = WCS(muse_hdu[map_type].header)
        muse_hdu.close()

        data_identifier = helper_func.SpecHelper.get_dap_data_identifier(res=res, ssp_model=ssp_model)

        self.muse_dap_map_data.update({
            'dap_map_data_%s_%s' % (data_identifier, map_type): muse_map_data,
            'dap_map_wcs_%s_%s' % (data_identifier, map_type): muse_map_wcs
        })

    def check_dap_map_loaded(self, res='copt', ssp_model='fiducial', map_type='HA6562_FLUX'):
        data_identifier = helper_func.SpecHelper.get_dap_data_identifier(res=res, ssp_model=ssp_model)
        if not 'dap_map_data_%s_%s' % (data_identifier, map_type) in self.muse_dap_map_data.keys():
            self.load_muse_dap_map(res=res, ssp_model=ssp_model, map_type=map_type)

    def get_muse_dap_map_cutout(self, ra_cutout, dec_cutout, cutout_size, map_type_list=None, res='copt', ssp_model='fiducial'):
        if map_type_list is None:
            map_type_list = ['HA6562_FLUX']
        elif isinstance(map_type_list, str):
            map_type_list = [map_type_list]

        cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')
        cutout_dict = {'cutout_pos': cutout_pos}
        cutout_dict.update({'cutout_size': cutout_size})
        cutout_dict.update({'map_type_list': map_type_list})

        data_identifier = helper_func.SpecHelper.get_dap_data_identifier(res=res, ssp_model=ssp_model)

        for map_type in map_type_list:
            # make sure that map typ is loaded
            self.check_dap_map_loaded(res=res, ssp_model=ssp_model, map_type=map_type)
            cutout_dict.update({
                '%s_%s_img_cutout' % (data_identifier, map_type):
                    helper_func.CoordTools.get_img_cutout(
                        img=self.muse_dap_map_data['dap_map_data_%s_%s' % (data_identifier, map_type)],
                        wcs=self.muse_dap_map_data['dap_map_wcs_%s_%s' % (data_identifier, map_type)],
                        coord=cutout_pos, cutout_size=cutout_size)})
        return cutout_dict

    def load_muse_cube(self, res='copt'):
        """

        Parameters
        ----------
        res : str

        """
        file_path = self.get_muse_data_file_name(data_prod='datacubes', res=res)
        # get MUSE data
        muse_hdu = fits.open(file_path)
        # get header
        hdr = muse_hdu['DATA'].header
        # get wavelength
        wave_muse = hdr['CRVAL3'] + np.arange(hdr['NAXIS3']) * hdr['CD3_3']
        # get data and variance cube
        data_cube_muse = muse_hdu['DATA'].data
        var_cube_muse = muse_hdu['STAT'].data
        # get WCS
        wcs_3d_muse = WCS(hdr)
        wcs_2d_muse = wcs_3d_muse.celestial

        muse_hdu.close()

        self.muse_datacube_data.update({
            'wave_%s' % res: wave_muse,
            'data_cube_%s' % res: data_cube_muse,
            'var_cube_%s' % res: var_cube_muse,
            'hdr_%s' % res: hdr,
            'wcs_3d_%s' % res: wcs_3d_muse,
            'wcs_2d_%s' % res: wcs_2d_muse

        })

    def load_kcwi_cube(self):
        """
        load kcwi cube into

        """
        file_path = self.get_kcwi_data_file_name()
        # get MUSE data
        kcwi_hdu = fits.open(file_path)

        # get header
        hdr = kcwi_hdu[0].header
        # get WCS
        wcs_3d_kcwi = WCS(hdr)
        wcs_2d_kcwi = wcs_3d_kcwi.celestial

        # get wavelength
        wave_kcwi = hdr['CRVAL3'] + np.arange(hdr['NAXIS3']) * hdr['CDELT3']

        # get data, variance and mask cube
        data_cube_kcwi = kcwi_hdu[0].data
        var_cube_kcwi = kcwi_hdu[1].data
        mask_cube_kcwi = kcwi_hdu[2].data

        kcwi_hdu.close()

        self.kcwi_datacube_data.update({
            'wave': wave_kcwi,
            'data_cube': data_cube_kcwi,
            'var_cube': var_cube_kcwi,
            'mask_cube': mask_cube_kcwi,
            'hdr': hdr,
            'wcs_3d': wcs_3d_kcwi,
            'wcs_2d': wcs_2d_kcwi

        })

    def get_kcwi_lsf(self, wave):
        """
        Adopted from van Dokkum+2019 2019ApJ...880...91V
        Equation 11

        Parameters
        ----------
        wave : float or array
            in Units of angstrom
        Return
        ---------
        lsf : floar or array
            in Units of angstrom

        """

        return 0.377 - 5.79e-5 * (wave - 5000) - 1.144e-7 * ((wave - 5000)**2)

    def get_muse_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of MUSE observations

        Returns
        -------
        coverage_mask : dict
        """
        with open(self.path2obs_cover_hull / ('%s_muse_obs_hull_dict.npy' % self.spec_target_name), 'rb') as file_name:
            return pickle.load(file_name)

    def get_kcwi_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of KCWI observations

        Returns
        -------
        coverage_mask : dict
        """
        with open(self.path2obs_cover_hull / ('%s_kcwi_obs_hull_dict.npy' % self.spec_target_name), 'rb') as file_name:
            return pickle.load(file_name)

    def check_coords_covered_by_muse(self, ra, dec, res='copt', max_dist_dist2hull_arcsec=2):
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

        band_hull_dict = self.get_muse_obs_coverage_hull_dict()[res]
        coverage_mask = np.zeros(len(ra), dtype=bool)
        hull_data_ra = np.array([])
        hull_data_dec = np.array([])

        for hull_idx in band_hull_dict.keys():
            ra_hull = band_hull_dict[hull_idx]['ra']
            dec_hull = band_hull_dict[hull_idx]['dec']
            hull_data_ra = np.concatenate([hull_data_ra, ra_hull])
            hull_data_dec = np.concatenate([hull_data_dec, dec_hull])
            coverage_mask += helper_func.GeometryTools.check_points_in_polygon(x_point=ra, y_point=dec,
                                                                               x_data_hull=ra_hull,
                                                                               y_data_hull=dec_hull)

        coverage_mask *= helper_func.GeometryTools.flag_close_points2ensemble(
            x_data=ra, y_data=dec, x_data_ensemble=hull_data_ra,y_data_ensemble=hull_data_dec,
            max_dist2ensemble=max_dist_dist2hull_arcsec/3600)

        return coverage_mask

    def check_coords_covered_by_kcwi(self, ra, dec, max_dist_dist2hull_arcsec=2):
        """
        Function to check if coordinate points are inside MUSE observation

        Parameters
        ----------
        ra : float or ``np.ndarray``
        dec : float or ``np.ndarray``
        max_dist_dist2hull_arcsec : float

        Returns
        -------
        coverage_dict : ``np.ndarray``
        """

        if isinstance(ra, float):
            ra = [ra]
            dec = [dec]

        band_hull_dict = self.get_kcwi_obs_coverage_hull_dict()
        coverage_mask = np.zeros(len(ra), dtype=bool)
        hull_data_ra = np.array([])
        hull_data_dec = np.array([])

        for hull_idx in band_hull_dict.keys():
            ra_hull = band_hull_dict[hull_idx]['ra']
            dec_hull = band_hull_dict[hull_idx]['dec']
            hull_data_ra = np.concatenate([hull_data_ra, ra_hull])
            hull_data_dec = np.concatenate([hull_data_dec, dec_hull])
            coverage_mask += helper_func.GeometryTools.check_points_in_polygon(x_point=ra, y_point=dec,
                                                                               x_data_hull=ra_hull,
                                                                               y_data_hull=dec_hull)

        coverage_mask *= helper_func.GeometryTools.flag_close_points2ensemble(
            x_data=ra, y_data=dec, x_data_ensemble=hull_data_ra,y_data_ensemble=hull_data_dec,
            max_dist2ensemble=max_dist_dist2hull_arcsec/3600)

        return coverage_mask

    def extract_muse_spec_circ_app(self, ra, dec, rad_arcsec, wave_range=None, res='copt'):
        """

        Parameters
        ----------
        ra, dec : float
        rad_arcsec : float
            radius in arcseconds
        wave_range : tuple or None
        res : str

        Return
        ---------
        spec_dict : dict
            Dictionary with all spectra related values
            fluxes are in units of ``10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$``
        """

        # make sure muse cube is loaded
        if 'data_cube_%s' % res not in self.muse_datacube_data.keys():
            self.load_muse_cube(res=res)

        # get select spectra from coordinates
        obj_coords_world = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        obj_coords_muse_pix = self.muse_datacube_data['wcs_2d_%s' % res].world_to_pixel(obj_coords_world)
        selection_radius_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=rad_arcsec, wcs=self.muse_datacube_data['wcs_2d_%s' % res])
        # select spaxels which are inside the selected radius
        x_lin_muse = np.linspace(1, self.muse_datacube_data['data_cube_%s' % res].shape[2],
                                 self.muse_datacube_data['data_cube_%s' % res].shape[2])
        y_lin_muse = np.linspace(1, self.muse_datacube_data['data_cube_%s' % res].shape[1],
                                 self.muse_datacube_data['data_cube_%s' % res].shape[1])
        x_data_muse, y_data_muse = np.meshgrid(x_lin_muse, y_lin_muse)
        mask_spectrum = (np.sqrt((x_data_muse - obj_coords_muse_pix[0]) ** 2 +
                                 (y_data_muse - obj_coords_muse_pix[1]) ** 2) < selection_radius_pix)

        # extract fluxes
        spec_flux = np.sum(self.muse_datacube_data['data_cube_%s' % res][:, mask_spectrum], axis=1)
        spec_flux_err = np.sqrt(np.sum(self.muse_datacube_data['var_cube_%s' % res][:, mask_spectrum], axis=1))
        # rescale to the unit of 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$
        spec_flux *= 1e-20
        spec_flux_err *= 1e-20
        # get wavelengths
        lam = self.muse_datacube_data['wave_%s' % res]
        # getting line spread function
        lsf = get_MUSE_polyFWHM(self.muse_datacube_data['wave_%s' % res], version="udf10")

        # get wavelength range
        if wave_range is None:
            lam_range = [np.nanmin(self.muse_datacube_data['wave_%s' % res][np.invert(np.isnan(spec_flux))]),
                         np.nanmax(self.muse_datacube_data['wave_%s' % res][np.invert(np.isnan(spec_flux))])]
        else:
            lam_range = wave_range
        # make sure wave length range is applied to all arrays
        mask_wave_range = (lam > lam_range[0]) & (lam < lam_range[1])
        spec_flux = spec_flux[mask_wave_range]
        spec_flux_err = spec_flux_err[mask_wave_range]
        lam = lam[mask_wave_range]
        lsf = lsf[mask_wave_range]
        good_pixel_mask = np.invert(np.isnan(spec_flux) + np.isinf(spec_flux))

        return {'lam_range': lam_range, 'spec_flux': spec_flux, 'spec_flux_err': spec_flux_err, 'lam': lam,
                'lsf': lsf, 'good_pixel_mask': good_pixel_mask, 'rad_arcsec': rad_arcsec}

    def extract_kcwi_spec_circ_app(self, ra, dec, rad_arcsec, wave_range=None):
        """

        Parameters
        ----------
        ra, dec : float
        rad_arcsec : float
            radius in arcseconds
        wave_range : tuple or None

        Return
        ---------
        spec_dict : dict
            Dictionary with all spectra related values
            fluxes are in units of ``10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$``
        """

        # make sure kcwi cube is loaded
        if 'data_cube' not in self.kcwi_datacube_data.keys():
            self.load_kcwi_cube()

        # get select spectra from coordinates
        obj_coords_world = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        obj_coords_kcwi_pix = self.kcwi_datacube_data['wcs_2d'].world_to_pixel(obj_coords_world)
        selection_radius_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=rad_arcsec, wcs=self.kcwi_datacube_data['wcs_2d'], dim=0)

        # select spaxels which are inside the selected radius
        x_lin_kcwi = np.linspace(1, self.kcwi_datacube_data['data_cube'].shape[2],
                                 self.kcwi_datacube_data['data_cube'].shape[2])
        y_lin_kcwi = np.linspace(1, self.kcwi_datacube_data['data_cube'].shape[1],
                                 self.kcwi_datacube_data['data_cube'].shape[1])
        x_data_kcwi, y_data_kcwi = np.meshgrid(x_lin_kcwi, y_lin_kcwi)
        mask_spectrum = (np.sqrt((x_data_kcwi - obj_coords_kcwi_pix[0]) ** 2 +
                                 (y_data_kcwi - obj_coords_kcwi_pix[1]) ** 2) < selection_radius_pix)

        # extract fluxes
        spec_flux = np.sum(self.kcwi_datacube_data['data_cube'][:, mask_spectrum], axis=1)
        spec_flux_err = np.sqrt(np.sum(self.kcwi_datacube_data['var_cube'][:, mask_spectrum], axis=1))
        # rescale to the unit of 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$
        spec_flux *= 1e-16
        spec_flux_err *= 1e-16
        # get wavelengths
        lam = self.kcwi_datacube_data['wave']
        # getting line spread function
        lsf = get_MUSE_polyFWHM(self.kcwi_datacube_data['wave'], version="udf10")

        # get wavelength range
        if wave_range is None:
            lam_range = [np.nanmin(self.kcwi_datacube_data['wave'][np.invert(np.isnan(spec_flux))]),
                         np.nanmax(self.kcwi_datacube_data['wave'][np.invert(np.isnan(spec_flux))])]
        else:
            lam_range = wave_range
        # make sure wave length range is applied to all arrays
        mask_wave_range = (lam > lam_range[0]) & (lam < lam_range[1])
        spec_flux = spec_flux[mask_wave_range]
        spec_flux_err = spec_flux_err[mask_wave_range]
        lam = lam[mask_wave_range]
        lsf = lsf[mask_wave_range]
        good_pixel_mask = np.invert(np.isnan(spec_flux) + np.isinf(spec_flux))

        return {'lam_range': lam_range, 'spec_flux': spec_flux, 'spec_flux_err': spec_flux_err, 'lam': lam,
                'lsf': lsf, 'good_pixel_mask': good_pixel_mask, 'rad_arcsec': rad_arcsec}

    def extract_kcwi_sub_cube(self, ra, dec, cutout_size):
        """

        Parameters
        ----------
        ra, dec : float
        cutout_size : tuple

        Return
        ---------
        spec_dict : dict
            Dictionary with all spectra related values
            fluxes are in units of ``10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$``
        """

        # make sure kcwi cube is loaded
        if 'data_cube' not in self.kcwi_datacube_data.keys():
            self.load_kcwi_cube()

        # get select spectra from coordinates
        obj_coords_world = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        obj_coords_kcwi_pix = self.kcwi_datacube_data['wcs_2d'].world_to_pixel(obj_coords_world)
        width_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=cutout_size[0], wcs=self.kcwi_datacube_data['wcs_2d'], dim=0)
        length_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=cutout_size[1], wcs=self.kcwi_datacube_data['wcs_2d'], dim=0)

        print(width_pix, length_pix)

        #get a sub cutout


        exit()


        selection_radius_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=rad_arcsec, wcs=self.kcwi_datacube_data['wcs_2d'], dim=0)

        # select spaxels which are inside the selected radius
        x_lin_kcwi = np.linspace(1, self.kcwi_datacube_data['data_cube'].shape[2],
                                 self.kcwi_datacube_data['data_cube'].shape[2])
        y_lin_kcwi = np.linspace(1, self.kcwi_datacube_data['data_cube'].shape[1],
                                 self.kcwi_datacube_data['data_cube'].shape[1])
        x_data_kcwi, y_data_kcwi = np.meshgrid(x_lin_kcwi, y_lin_kcwi)
        mask_spectrum = (np.sqrt((x_data_kcwi - obj_coords_kcwi_pix[0]) ** 2 +
                                 (y_data_kcwi - obj_coords_kcwi_pix[1]) ** 2) < selection_radius_pix)

        # extract fluxes
        spec_flux = np.sum(self.kcwi_datacube_data['data_cube'][:, mask_spectrum], axis=1)
        spec_flux_err = np.sqrt(np.sum(self.kcwi_datacube_data['var_cube'][:, mask_spectrum], axis=1))
        # rescale to the unit of 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$
        spec_flux *= 1e-16
        spec_flux_err *= 1e-16
        # get wavelengths
        lam = self.kcwi_datacube_data['wave']
        # getting line spread function
        lsf = get_MUSE_polyFWHM(self.kcwi_datacube_data['wave'], version="udf10")

        # get wavelength range
        if wave_range is None:
            lam_range = [np.nanmin(self.kcwi_datacube_data['wave'][np.invert(np.isnan(spec_flux))]),
                         np.nanmax(self.kcwi_datacube_data['wave'][np.invert(np.isnan(spec_flux))])]
        else:
            lam_range = wave_range
        # make sure wave length range is applied to all arrays
        mask_wave_range = (lam > lam_range[0]) & (lam < lam_range[1])
        spec_flux = spec_flux[mask_wave_range]
        spec_flux_err = spec_flux_err[mask_wave_range]
        lam = lam[mask_wave_range]
        lsf = lsf[mask_wave_range]
        good_pixel_mask = np.invert(np.isnan(spec_flux) + np.isinf(spec_flux))

        return {'lam_range': lam_range, 'spec_flux': spec_flux, 'spec_flux_err': spec_flux_err, 'lam': lam,
                'lsf': lsf, 'good_pixel_mask': good_pixel_mask, 'rad_arcsec': rad_arcsec}

    def extract_muse_spec_gauss_app(self, ra, dec, fwhm_arcsec, wave_range=None, res='copt'):
        """

        Parameters
        ----------
        ra, dec : float
        rad_arcsec : float
            radius in arcseconds
        wave_range : tuple or None
        res : str

        Return
        ---------
        spec_dict : dict
            Dictionary with all spectra related values
            fluxes are in units of ``10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$``
        """

        # make sure muse cube is loaded
        if 'data_cube_%s' % res not in self.muse_datacube_data.keys():
            self.load_muse_cube(res=res)

        # get select spectra from coordinates
        obj_coords_world = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        obj_coords_muse_pix = self.muse_datacube_data['wcs_2d_%s' % res].world_to_pixel(obj_coords_world)
        selection_fwhm_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=fwhm_arcsec, wcs=self.muse_datacube_data['wcs_2d_%s' % res])
        selection_sig_pix = helper_func.TransTools.gauss_fwhm2sig(fwhm=selection_fwhm_pix)

        # select spaxels which are inside the selected radius
        x_lin_muse = np.linspace(1, self.muse_datacube_data['data_cube_%s' % res].shape[2],
                                 self.muse_datacube_data['data_cube_%s' % res].shape[2])
        y_lin_muse = np.linspace(1, self.muse_datacube_data['data_cube_%s' % res].shape[1],
                                 self.muse_datacube_data['data_cube_%s' % res].shape[1])
        x_data_muse, y_data_muse = np.meshgrid(x_lin_muse, y_lin_muse)
        # create 2d Gauss
        gauss = helper_func.FuncAndModels.gauss2d_rot(x=x_data_muse, y=y_data_muse, amp=1, x0=obj_coords_muse_pix[0],
                                                      y0=obj_coords_muse_pix[1], sig_x=selection_sig_pix, sig_y=selection_sig_pix, theta=0)

        scaled_spec = self.muse_datacube_data['data_cube_%s' % res] * gauss
        scaled_var = self.muse_datacube_data['var_cube_%s' % res] * gauss

        # import matplotlib.pyplot as plt
        #
        # plt.imshow(np.nansum(scaled_spec, axis=0))
        # plt.show()
        # exit()

        # now to accelerate this we select only those spaxels which are somwhere around this
        mask_spectrum = (np.sqrt((x_data_muse - obj_coords_muse_pix[0]) ** 2 +
                                 (y_data_muse - obj_coords_muse_pix[1]) ** 2) < 10 * selection_fwhm_pix)
        # extract fluxes
        spec_flux = np.nansum(scaled_spec[:, mask_spectrum], axis=1)
        spec_flux_err = np.sqrt(np.nansum(scaled_var[:, mask_spectrum], axis=1))





        # spec_flux = np.nansum(self.muse_datacube_data['data_cube_%s' % res] * gauss, axis=(1, 2))
        # spec_flux_err = np.sqrt(np.nansum(self.muse_datacube_data['var_cube_%s' % res] * gauss, axis=(1, 2)))

        # rescale to the unit of 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$
        spec_flux *= 1e-20
        spec_flux_err *= 1e-20
        # get wavelengths
        lam = self.muse_datacube_data['wave_%s' % res]
        # getting line spread function
        lsf = get_MUSE_polyFWHM(self.muse_datacube_data['wave_%s' % res], version="udf10")

        # get wavelength range
        if wave_range is None:
            lam_range = [np.nanmin(self.muse_datacube_data['wave_%s' % res][np.invert(np.isnan(spec_flux))]),
                         np.nanmax(self.muse_datacube_data['wave_%s' % res][np.invert(np.isnan(spec_flux))])]
        else:
            lam_range = wave_range
        # make sure wave length range is applied to all arrays
        mask_wave_range = (lam > lam_range[0]) & (lam < lam_range[1])
        spec_flux = spec_flux[mask_wave_range]
        spec_flux_err = spec_flux_err[mask_wave_range]
        lam = lam[mask_wave_range]
        lsf = lsf[mask_wave_range]
        good_pixel_mask = np.invert(np.isnan(spec_flux) + np.isinf(spec_flux))

        return {'lam_range': lam_range, 'spec_flux': np.array(spec_flux), 'spec_flux_err': np.array(spec_flux_err), 'lam': np.array(lam),
                'lsf': np.array(lsf), 'good_pixel_mask': np.array(good_pixel_mask), 'fwhm_arcsec': fwhm_arcsec}

