"""
Construct a data access structure for HST and JWST imaging data
"""
import os.path
from pathlib import Path
import pickle

import numpy as np
from astropy.extern.ply.yacc import VersionError
from scipy.constants import c as speed_of_light

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning

from astroquery.skyview import SkyView
from phangs_data_access import phangs_access_config, helper_func, phot_tools, phangs_info, phys_params

# ignore JWST pipline warning which comes from a header modification
# see also https://github.com/astropy/astropy/issues/13463
import warnings
warnings.simplefilter("ignore", category=FITSFixedWarning)


class PhotAccess:
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """

    def __init__(self, phot_target_name=None, phot_hst_target_name=None, phot_hst_ha_cont_sub_target_name=None,
                 phot_nircam_target_name=None, phot_miri_target_name=None, phot_astrosat_target_name=None,
                 nircam_data_ver='v1p1p1', miri_data_ver='v1p1p1', astrosat_data_ver = 'v1p0'):
        """

        Parameters
        ----------
        phot_target_name : str
        """

        # check if the target is in the list specifying the access pipeline
        if (
                # check if the target is an HST target
                (phot_target_name not in phangs_info.full_hst_galaxy_list) &

                # check if the target is an JWST target
                (phot_target_name not in phangs_info.full_jwst_galaxy_list) &

                # check if the target is an ASTROSAT target
                (phot_target_name not in phangs_info.full_astrosat_galaxy_list) &

                # there is always the possibility to provide None
                (phot_target_name is not None)):
            raise AttributeError('The target %s is not in the PHANGS photometric sample or has not been added to '
                                 'the current package version' % phot_target_name)

        # load target names into constructor to specify data access
        if phot_target_name is not None:
            self.phot_target_name = helper_func.FileTools.target_name_no_directions(target=phot_target_name)
        else:
            self.phot_target_name = None
        # now specific target names for different telescopes as sometimes their names and directions can vary
        # hst
        if phot_hst_target_name is not None:  self.phot_hst_target_name = phot_hst_target_name
        elif (phot_hst_target_name is None) & (phot_target_name is not None):
            self.phot_hst_target_name = phot_target_name
        else: self.phot_hst_target_name = None
        # hst_ha_cont_sub
        if phot_hst_ha_cont_sub_target_name is not None:
            self.phot_hst_ha_cont_sub_target_name = phot_hst_ha_cont_sub_target_name
        elif (phot_hst_ha_cont_sub_target_name is None) & (phot_target_name is not None):
            self.phot_hst_ha_cont_sub_target_name = phot_target_name
        else: self.phot_hst_ha_cont_sub_target_name = None
        # nircam
        if phot_nircam_target_name is not None:  self.phot_nircam_target_name = phot_nircam_target_name
        elif (phot_nircam_target_name is None) & (phot_target_name is not None):
            self.phot_nircam_target_name = helper_func.FileTools.target_name_no_directions(target=phot_target_name)
        else: self.phot_nircam_target_name = None
        # miri
        if phot_miri_target_name is not None: self.phot_miri_target_name = phot_miri_target_name
        elif (phot_miri_target_name is None) & (phot_target_name is not None):
            self.phot_miri_target_name = helper_func.FileTools.target_name_no_directions(target=phot_target_name)
        else: self.phot_miri_target_name = None
        # astrosat
        if phot_astrosat_target_name is not None: self.phot_astrosat_target_name = phot_astrosat_target_name
        elif (phot_astrosat_target_name is None) & (phot_target_name is not None):
            self.phot_astrosat_target_name = helper_func.FileTools.target_name_no_directions(target=phot_target_name)
        else: self.phot_astrosat_target_name = None

        # data versions can sometimes vary from target to target therefore we load them into the constructor
        self.nircam_data_ver = nircam_data_ver
        self.miri_data_ver = miri_data_ver
        self.astrosat_data_ver = astrosat_data_ver

        # loaded data dictionaries
        self.hst_bands_data = {}
        self.hst_ha_cont_sub_bands_data = {}
        self.nircam_bands_data = {}
        self.miri_bands_data = {}
        self.astrosat_bands_data = {}

        # get path to observation coverage hulls
        self.path2obs_cover_hull = (Path(__file__).parent.parent.absolute() / 'meta_data' / 'obs_coverage' /
                                    'data_output')

        super().__init__()

    def get_hst_img_file_name(self, band, file_type='sci'):
        """

        Parameters
        ----------
        band : str
        file_type : str
            can be sci, err or wht
        Returns
        -------
        data_file_path : ``Path``
        """

        if band in phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['acs']:
            instrument = 'acs'
        elif band in phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['uvis']:
            instrument = 'uvis'
        elif band in phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['acs_uvis']:
            instrument = 'acs_uvis'
        elif band in phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['ir']:
            instrument = 'ir'
        else:
            raise KeyError(band, ' is not observed by HST for the target ', self.phot_hst_target_name)

        if self.phot_hst_target_name == 'ngc5194':
            hst_data_folder = (Path(phangs_access_config.phangs_config_dict['hst_data_path']) /
                               self.phot_hst_target_name)

        else:

            hst_data_folder = (Path(phangs_access_config.phangs_config_dict['hst_data_path']) / 'HST_reduced_images' /
                               helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_target_name) /
                               (instrument + band.lower()))

        if file_type in ['sci', 'wht']:
            if self.phot_hst_target_name == 'ngc5194':
                if instrument == 'uvis': instrument_str = 'WFC3_UVIS'; ending = 'drc'
                elif instrument == 'acs': instrument_str = 'ACS_WFC'; ending = 'drc'
                elif instrument == 'ir': instrument_str = 'WFC3_IR'; ending = 'drz'
                else: raise KeyError('instrument is not understood in this context?')
                file_name = '%s_%s_%s_%s.fits' % (band.upper(), self.phot_hst_target_name.upper(), instrument_str, ending)
                # file_name = '%s_HST_%s_IVM_%s.fits' % (band.upper(), instrument_str, ending)

            else:
                file_name = '%s_%s_%s_exp_drc_%s.fits' % (
                    helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_target_name),
                    instrument, band.lower(), file_type)
        elif file_type == 'err':
            if self.phot_hst_target_name == 'ngc5194':
                if instrument == 'uvis': instrument_str = 'WFC3_UVIS'; ending='drc'
                elif instrument == 'acs': instrument_str = 'ACS_WFC'; ending='drc'
                elif instrument == 'ir': instrument_str = 'WFC3_IR'; ending='drz'
                else: raise KeyError('instrument is not understood in this context?')
                file_name = '%s_HST_%s_IVM_%s.fits' % (band.upper(), instrument_str, ending)
            else:
                file_name = '%s_%s_%s_%s_drc_wht.fits' % (helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_target_name),
                                                          instrument, band.lower(), file_type)
        else:
            raise KeyError('file_type must be in sci, err or wht')

        return Path(hst_data_folder) / file_name

    def get_hst_ha_cont_sub_img_file_name(self):
        """
        hst H-alpha continuum subtracted observation

        Returns
        -------
        data_file_path : ``Path``
        """

        if self.phot_hst_ha_cont_sub_target_name not in phangs_info.hst_ha_cont_sub_dict.keys():
            raise LookupError(self.phot_hst_ha_cont_sub_target_name, ' has no H-alpha observation ')

        if self.phot_hst_ha_cont_sub_target_name == 'ngc5194':
            hst_data_folder = Path(phangs_access_config.phangs_config_dict['hst_ha_cont_sub_data_path'])
        else:
            hst_data_folder = (Path(phangs_access_config.phangs_config_dict['hst_ha_cont_sub_data_path']) /
                               phangs_access_config.phangs_config_dict['hst_ha_cont_sub_ver'])

        if self.phot_hst_ha_cont_sub_target_name == 'ngc5194':
            file_name = 'F658N_acs_cont_sub_from_f555w_f814w.fits'
        else:
            if os.path.isfile(hst_data_folder /
                              ('%s_hst_ha.fits' % helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_ha_cont_sub_target_name))):
                file_name = '%s_hst_ha.fits' % helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_ha_cont_sub_target_name)
            elif os.path.isfile(hst_data_folder / ('%s_hst_%s_contsub.fits' %
                                                   (helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_ha_cont_sub_target_name),
                                                    helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name)))):
                file_name = ('%s_hst_%s_contsub.fits' %
                             (helper_func.FileTools.target_names_no_zeros(target=self.phot_hst_ha_cont_sub_target_name),
                              helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name)))
            else:
                raise KeyError('No H-alpha continuum subtracted product found for ', self.phot_hst_ha_cont_sub_target_name)

        return Path(hst_data_folder) / file_name

    def get_jwst_img_file_name(self, instrument, band):
        """

        Parameters
        ----------
        instrument : str
        band : str

        Returns
        -------
        data_file_path : Path
        """
        target_name = getattr(self, 'phot_%s_target_name' % instrument)

        data_folder = (Path(phangs_access_config.phangs_config_dict['%s_data_path' % instrument]) /
                       getattr(self, '%s_data_ver' % instrument) /
                       target_name)

        if target_name == 'ngc1068': extension = ''
        elif (target_name == 'ngc5194') & (instrument == 'miri'): extension = ''
        else: extension = '_anchor'

        if os.path.isfile(Path(data_folder) / Path('%s_%s_lv3_%s_i2d_align_v1p1p1_ff+rscd.fits' % (target_name, instrument, band.lower()))):
            file_name = '%s_%s_lv3_%s_i2d_align_v1p1p1_ff+rscd.fits' % (target_name, instrument, band.lower())
        else:
            file_name = '%s_%s_lv3_%s_i2d%s.fits' % (target_name, instrument, band.lower(), extension)

        return Path(data_folder) / Path(file_name)

    def get_astrosat_img_file_name(self, band):
        """

        Parameters
        ----------
        band : str
        Returns
        -------
        data_file_path : Path
        """

        astrosat_data_folder = (Path(phangs_access_config.phangs_config_dict['astrosat_data_path']) /
                                self.astrosat_data_ver /
                                'release')

        file_name = '%s_%s_bkg_subtracted_mw_corrected.fits' % (self.phot_astrosat_target_name.upper(),
                                                                band[:-1].upper())

        return Path(astrosat_data_folder) / Path(file_name)

    def load_hst_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str
        """

        # check if band is already loaded.
        # If so skip the loading and only change the units!
        if not ('%s_data_img' % band) in self.hst_bands_data.keys():

            # load the band observations
            if file_name is None:
                file_name = self.get_hst_img_file_name(band=band)

            if self.phot_hst_target_name == 'ngc5194': hdu_number = 'SCI'
            else: hdu_number = 0

            img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number=hdu_number)
            # rescale image to needed unit
            if band in list(phys_params.hst_wfc3_ir_bands_wave.keys()):
                img_data *= helper_func.UnitTools.get_hst_ir_img_conv_fct(band=band, img_header=img_header, img_wcs=img_wcs,
                                                                   flux_unit=flux_unit)
            else:
                img_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                                       flux_unit=flux_unit)
            self.hst_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                        '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                        '%s_pixel_area_size_sr_img' % band:
                                            img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        else:
            # data is already loaded so we have to change the units
            self.change_band_unit(band=band, new_unit=flux_unit)
        if load_err:
            # here it is important to note that the change unit operation also changes the unit of the error
            if not ('%s_data_err' % band) in self.hst_bands_data.keys():
                err_file_name = self.get_hst_img_file_name(band=band, file_type='err')
                if self.phot_hst_target_name == 'ngc5194':
                    err_hdu_number = 'WHT'
                    hdu_number = 'SCI'
                else:
                    err_hdu_number = 0
                    hdu_number = 0
                err_data_array, err_header, err_wcs = helper_func.FileTools.load_img(file_name=err_file_name, hdu_number=err_hdu_number)

                # it has to be said that in the PHANGS pipeline there are not the needed keywords in the header thus
                # the image has to be loaded as well here:
                if file_name is None:
                    file_name = self.get_hst_img_file_name(band=band)
                img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number=hdu_number)

                # use the inverse sqrt and exclude zeros
                mask_bad_err = err_data_array == 0
                err_data = np.zeros(err_data_array.shape)
                err_data[np.invert(mask_bad_err)] = 1/ np.sqrt(err_data_array[np.invert(mask_bad_err)])
                err_data[mask_bad_err] = np.nan

                # rescale image to needed unit

                if band in list(phys_params.hst_wfc3_ir_bands_wave.keys()):
                    err_data *= helper_func.UnitTools.get_hst_ir_img_conv_fct(band=band, img_header=img_header, img_wcs=img_wcs,
                                                                       flux_unit=flux_unit)
                else:
                    err_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                                           flux_unit=flux_unit)
                self.hst_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                            '%s_wcs_err' % band: err_wcs, '%s_unit_err' % band: flux_unit,
                                            '%s_pixel_area_size_sr_err' % band:
                                                img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_hst_ha_cont_sub_band(self, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        load_err : bool
        flux_unit : str
        file_name : str
        """
        band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name)
        # check if band is already loaded.
        # If so skip the loading and only change the units!
        if not ('%s_cont_sub_data_img' % band) in self.hst_ha_cont_sub_bands_data.keys():

            # load the band observations
            if file_name is None:
                file_name = self.get_hst_ha_cont_sub_img_file_name()

            img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

            # rescale image to needed unit
            if self.phot_hst_target_name == 'ngc5194':
                no_header_conv = True
            else:
                no_header_conv = False
            img_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                                   flux_unit=flux_unit, no_header_conv=no_header_conv)

            self.hst_ha_cont_sub_bands_data.update({'%s_cont_sub_data_img' % band: img_data,
                                                    '%s_cont_sub_header_img' % band: img_header,
                                                    '%s_cont_sub_wcs_img' % band: img_wcs,
                                                    '%s_cont_sub_unit_img' % band: flux_unit,
                                                    '%s_cont_sub_pixel_area_size_sr_img' % band:
                                                        (img_wcs.proj_plane_pixel_area().value *
                                                         phys_params.sr_per_square_deg)})
        else:
            # data is already loaded so we have to change the units
            self.change_band_unit(band=band, new_unit=flux_unit)
        if load_err:
            # TO DO: get errors !
            warnings.warn('Instead of continuum subtracted uncertainty the only the '
                          'H-alpha uncertainty will be loaded! This needs to change in future!')
            # raise NotImplementedError('Uncertainties are not yet available for HST H-alpha observations')
            # 3 just load h-alpha file error
            if self.phot_hst_target_name == 'ngc5194': hdu_number = 'SCI'
            else: hdu_number = 0
            err_file_name = self.get_hst_img_file_name(
                band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name),
                file_type='err')
            img_file_name = self.get_hst_img_file_name(
                band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name))
            img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=img_file_name, hdu_number=hdu_number)

            err_data_array, err_header, err_wcs = helper_func.FileTools.load_img(file_name=err_file_name, hdu_number=hdu_number)

            # use the inverse sqrt and exclude zeros
            mask_bad_err = err_data_array == 0
            err_data = np.zeros(err_data_array.shape)
            err_data[np.invert(mask_bad_err)] = 1 / np.sqrt(err_data_array[np.invert(mask_bad_err)])
            err_data[mask_bad_err] = np.nan

            # rescale image to needed unit
            err_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                                   flux_unit=flux_unit)

            self.hst_ha_cont_sub_bands_data.update({
                '%s_cont_sub_data_err' % band: err_data, '%s_cont_sub_header_err' % band: err_header,
                '%s_cont_sub_wcs_err' % band: err_wcs, '%s_cont_sub_unit_err' % band: flux_unit,
                '%s_cont_sub_pixel_area_size_sr_err' % band:
                    img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_nircam_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str

        """
        # load the band observations
        if file_name is None:
            file_name = self.get_jwst_img_file_name(instrument='nircam', band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='SCI')

        img_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)

        self.nircam_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                       '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                       '%s_pixel_area_size_sr_img' % band:
                                           img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            err_data, err_header, err_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='ERR')
            err_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)
            self.nircam_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                           '%s_wcs_err' % band: img_wcs, '%s_unit_err' % band: flux_unit,
                                           '%s_pixel_area_size_sr_err' % band:
                                               img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_miri_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str

        """
        # load the band observations
        if file_name is None:
            file_name = self.get_jwst_img_file_name(instrument='miri', band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='SCI')

        img_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)

        self.miri_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                     '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                     '%s_pixel_area_size_sr_img' % band:
                                         img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            err_data, err_header, err_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='ERR')

            err_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)
            self.miri_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                         '%s_wcs_err' % band: img_wcs, '%s_unit_err' % band: flux_unit,
                                         '%s_pixel_area_size_sr_err' % band:
                                             img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_astrosat_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str
        """
        # load the band observations
        if file_name is None:
            file_name = self.get_astrosat_img_file_name(band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

        conversion_factor = helper_func.UnitTools.get_astrosat_conv_fact(img_wcs=img_wcs, band=band,
                                                                         flux_unit=flux_unit)
        img_data *= conversion_factor

        self.astrosat_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                         '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                         '%s_pixel_area_size_sr_img' % band:
                                             img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            # the uncertainties are estimated to be approximately 5% in Hassani+2024 2024ApJS..271....2H
            self.astrosat_bands_data.update({'%s_data_err' % band: img_data * 0.05, '%s_header_err' % band: img_header,
                                             '%s_wcs_err' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                             '%s_pixel_area_size_sr_err' % band:
                                                 img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def get_phangs_obs_band_list(self):
        """
        assemble all observed photometry bands
        """
        band_list = []
        # HST band images (including H-alpha)
        if helper_func.ObsTools.check_hst_obs(target=self.phot_hst_target_name):
            band_list += helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name)
        # in continuum subtracted H-alpha obs is available also load it:
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name):
            hst_ha_band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name)
            band_list += [hst_ha_band + '_cont_sub']
        # nircam
        if helper_func.ObsTools.check_nircam_obs(target=self.phot_nircam_target_name, version=self.nircam_data_ver):
            band_list += helper_func.ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                       version=self.nircam_data_ver)
        # miri
        if helper_func.ObsTools.check_miri_obs(target=self.phot_miri_target_name, version=self.miri_data_ver):
            band_list += helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                     version=self.miri_data_ver)
        # astrosat
        if helper_func.ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
            band_list += helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name)

        return band_list

    def get_covered_hst_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2):
        if self.phot_hst_target_name in phangs_info.hst_obs_band_dict.keys():
            prelim_hst_broad_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name)
        else:
            prelim_hst_broad_band_list = []
        # check if band is really covered
        hst_broad_band_list = []
        for band in prelim_hst_broad_band_list:
            if self.check_coords_covered_by_band(telescope='hst', ra=ra, dec=dec, band=band, max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                hst_broad_band_list.append(band)
        return hst_broad_band_list

    def get_covered_hst_broad_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2):
        if self.phot_hst_target_name in phangs_info.hst_obs_band_dict.keys():
            prelim_hst_broad_band_list = helper_func.ObsTools.get_hst_obs_broad_band_list(target=self.phot_hst_target_name)
        else:
            prelim_hst_broad_band_list = []
        # check if band is really covered
        hst_broad_band_list = []
        for band in prelim_hst_broad_band_list:
            if self.check_coords_covered_by_band(telescope='hst', ra=ra, dec=dec, band=band, max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                hst_broad_band_list.append(band)
        return hst_broad_band_list

    def get_covered_hst_ha_band(self, ra, dec, max_dist_dist2hull_arcsec=2):
        if self.phot_hst_target_name in phangs_info.hst_obs_band_dict.keys():
            hst_ha_band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name)
        else:
            return False
        # check if band is really covered
        if self.check_coords_covered_by_band(telescope='hst', ra=ra, dec=dec, band=hst_ha_band, max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
            return hst_ha_band

    def get_covered_hst_ha_cont_sub_band(self, ra, dec, max_dist_dist2hull_arcsec=2):

        # same for H- alpha observations
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            hst_ha_band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name)
            if not self.check_coords_covered_by_band(telescope='hst', ra=ra, dec=dec, band=hst_ha_band,
                                                     max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                hst_ha_band = None
        else:
            hst_ha_band = None
        return hst_ha_band

    def get_covered_nircam_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2):
        # nircam
        if helper_func.ObsTools.check_nircam_obs(target=self.phot_nircam_target_name, version=self.nircam_data_ver):
            prelim_nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                                    version=self.nircam_data_ver)
        else:
            prelim_nircam_band_list = []
        # make sure all bands are covered
        nircam_band_list = []
        for band in prelim_nircam_band_list:
            if self.check_coords_covered_by_band(telescope='nircam', ra=ra, dec=dec, band=band,
                                                 max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                nircam_band_list.append(band)
        return nircam_band_list

    def get_covered_miri_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2):
        # miri
        if helper_func.ObsTools.check_miri_obs(target=self.phot_miri_target_name, version=self.miri_data_ver):
            prelim_miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                                version=self.miri_data_ver)
        else:
            prelim_miri_band_list = []
        # make sure all bands are covered
        miri_band_list = []
        for band in prelim_miri_band_list:
            if self.check_coords_covered_by_band(telescope='miri', ra=ra, dec=dec, band=band,
                                                 max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                miri_band_list.append(band)
        return miri_band_list

    def get_covered_astrosat_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2):
        if helper_func.ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
            prelim_astrosat_band_list = helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name)
        else:
            prelim_astrosat_band_list = []
        # make sure all bands are covered
        astrosat_band_list = []
        for band in prelim_astrosat_band_list:
            if self.check_coords_covered_by_band(obs='astrosat', ra=ra, dec=dec, band=band,
                                                 max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec):
                astrosat_band_list.append(band)
        return astrosat_band_list

    def get_covered_phangs_band_list(self, ra, dec, max_dist_dist2hull_arcsec=2, hst_broad_band=True, hst_ha=True,
                                     hst_ha_cont_sub=True,  nircam=True, miri=True, astrosat=True):

        band_list = []
        if hst_broad_band:
            band_list += self.get_covered_hst_broad_band_list(ra=ra, dec=dec,
                                                              max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)
        if hst_ha:
            band_list += [self.get_covered_hst_ha_band(ra=ra, dec=dec,
                                                       max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)]
        if hst_ha_cont_sub:
            band_list += [self.get_covered_hst_ha_band(ra=ra, dec=dec,
                                                       max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec) +
                          '_cont_sub']
        if nircam:
            band_list += self.get_covered_nircam_band_list(ra=ra, dec=dec,
                                                           max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)
        if miri:
            band_list += self.get_covered_miri_band_list(ra=ra, dec=dec,
                                                         max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)
        if astrosat:
            band_list += self.get_covered_astrosat_band_list(ra=ra, dec=dec,
                                                             max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)

        return band_list

    def load_phangs_bands(self, band_list=None, flux_unit='Jy', load_err=False, load_hst=True, load_hst_ha=True,
                          load_nircam=True, load_miri=True, load_astrosat=True):
        """
        wrapper to load all available HST, HST-H-alpha, NIRCAM, MIRI and Astrosat observations into the constructor
        This function checks if the band is already loaded and skips the loading if this is the case.
        If furthermore also adapts the unit in case another unit is required.

        Parameters
        ----------
        band_list : list or str
        flux_unit : str
        load_err : bool
        load_hst: bool
        load_hst_ha: bool
        load_nircam: bool
        load_miri: bool
        load_astrosat: bool

        """
        # if only one band should be loaded
        if isinstance(band_list, str):
            band_list = [band_list]
        # if band list is none we get a list with all observed bands in order of wavelength
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        # load bands
        for band in band_list:
            # load flag indicates if a band was loaded or nothing matching was found
            band_loaded_flag = False
            # check hst
            # first check if object has HST observation
            if self.phot_hst_target_name in phangs_info.hst_obs_band_dict.keys():
                if band in (phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['acs'] +
                            phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['uvis'] +
                            phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['acs_uvis'] +
                            phangs_info.hst_obs_band_dict[self.phot_hst_target_name]['ir']):
                    band_loaded_flag = True
                    # check if band is already loaded
                    if ((('%s_data_img' % band) not in self.hst_bands_data) |
                            ((('%s_data_err' % band) not in self.hst_bands_data) & load_err)):
                        if load_hst:
                            self.load_hst_band(band=band, flux_unit=flux_unit, load_err=load_err)
                    else:
                        # make sure it has the correct unit
                        self.change_band_unit(band=band, new_unit=flux_unit)
                        continue
            # check hst H-alpha
            if self.phot_hst_ha_cont_sub_target_name in phangs_info.hst_ha_cont_sub_dict.keys():
                # check hst H-alpha continuum subtracted
                # check hst H-alpha
                if band in ['F657N_cont_sub', 'F658N_cont_sub']:
                    band_loaded_flag = True
                    # check if band is already loaded
                    if (('%s_data_img' % band not in self.hst_ha_cont_sub_bands_data) |
                            (('%s_data_err' % band not in self.hst_ha_cont_sub_bands_data) & load_err)):
                        if load_hst_ha:
                            self.load_hst_ha_cont_sub_band(flux_unit=flux_unit, load_err=load_err)
                    else:
                        # make sure it has the correct unit
                        self.change_band_unit(band=band, new_unit=flux_unit)
                        continue
            # check nircam
            if self.phot_nircam_target_name in getattr(phangs_info, 'jwst_obs_band_dict_%s' % self.nircam_data_ver).keys():
                if band in getattr(phangs_info, 'jwst_obs_band_dict_%s' % self.nircam_data_ver)[self.phot_nircam_target_name]['nircam_observed_bands']:
                    band_loaded_flag = True
                    # check if band is already loaded
                    if ((('%s_data_img' % band) not in self.nircam_bands_data) |
                            ((('%s_data_err' % band) not in self.nircam_bands_data) & load_err)):
                        if load_nircam:
                            self.load_nircam_band(band=band, flux_unit=flux_unit, load_err=load_err)
                    else:
                        continue
            # check miri
            if self.phot_miri_target_name in getattr(phangs_info, 'jwst_obs_band_dict_%s' % self.miri_data_ver).keys():
                if band in getattr(phangs_info, 'jwst_obs_band_dict_%s' % self.miri_data_ver)[self.phot_miri_target_name]['miri_observed_bands']:
                    band_loaded_flag = True
                    # check if band is already loaded
                    if ((('%s_data_img' % band) not in self.miri_bands_data) |
                            ((('%s_data_err' % band) not in self.miri_bands_data) & load_err)):
                        if load_miri:
                            self.load_miri_band(band=band, flux_unit=flux_unit, load_err=load_err)
                    else:
                        continue
            # check astrosat
            if self.phot_astrosat_target_name in getattr(phangs_info, 'astrosat_obs_band_dict_%s' % self.astrosat_data_ver).keys():
                if band in getattr(phangs_info, 'astrosat_obs_band_dict_%s' % self.astrosat_data_ver)[self.phot_astrosat_target_name]['observed_bands']:
                    band_loaded_flag = True
                    # check if band is already loaded
                    if ((('%s_data_img' % band) not in self.astrosat_bands_data) |
                            ((('%s_data_err' % band) not in self.astrosat_bands_data) & load_err)):
                        if load_astrosat:
                            self.load_astrosat_band(band=band, flux_unit=flux_unit, load_err=load_err)
                    else:
                        # make sure it has the correct unit
                        self.change_band_unit(band=band, new_unit=flux_unit)
                        continue
            if not band_loaded_flag:
                raise KeyError('Band is not found in possible band lists')

    def change_phangs_band_units(self, band_list=None, new_unit='MJy/sr'):
        """

        Parameters
        ----------
        band_list : list
        new_unit : str
        """
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        for band in band_list:
            # check if band was loaded!
            if ('%s_data_img' % band) in (list(self.hst_bands_data.keys()) +
                                          list(self.hst_ha_cont_sub_bands_data.keys()) +
                                          list(self.nircam_bands_data.keys()) + list(self.miri_bands_data.keys()) +
                                          list(self.miri_bands_data.keys())):
                self.change_band_unit(band=band, new_unit=new_unit)

    def change_band_unit(self, band, new_unit='MJy/sr'):
        """
        will change loaded data to the needed unit. This will directly change all data saved in the constructor
        Parameters
        ----------
        band : str
        new_unit : str
            this can be :
            'mJy', 'Jy', 'MJy/sr' or 'erg A-1 cm-2 s-1'
        """
        # first we need to make sure what was the old unit and for which instrument.
        # Furthermore, we need the wavelength for some transformations
        if band in helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name):
            instrument = 'hst'
            band_wave = (
                helper_func.ObsTools.get_hst_band_wave(band=band, instrument=helper_func.ObsTools.get_hst_instrument(
                    target=self.phot_hst_target_name, band=band), unit='angstrom'))
        elif band == (helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name) + '_cont_sub'):
            instrument = 'hst_ha_cont_sub'
            band_wave = (
                helper_func.ObsTools.get_hst_band_wave(
                    band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name),
                    instrument=helper_func.ObsTools.get_hst_ha_instrument(target=self.phot_hst_ha_cont_sub_target_name),
                    unit='angstrom'))
        elif band in helper_func.ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                   version=self.nircam_data_ver):
            instrument = 'nircam'
            band_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, unit='angstrom')
        elif band in helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                 version=self.miri_data_ver):
            instrument = 'miri'
            band_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri', unit='angstrom')

        elif band in helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name):
            instrument = 'astrosat'
            band_wave = helper_func.ObsTools.get_astrosat_band_wave(band=band, unit='angstrom')
        else:
            raise KeyError('the band <%s> is not under the observed bands!' % band)

        # now we create a conversion factor
        # get the old unit
        old_unit = getattr(self, '%s_bands_data' % instrument)['%s_unit_img' % band]
        # get also pixel sizes
        pixel_size = getattr(self, '%s_bands_data' % instrument)['%s_pixel_area_size_sr_img' % band]
        # check if units are in the list of possible transformations
        # assert old_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']
        # assert new_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']
        #
        # conversion_factor = 1
        # if old_unit != new_unit:
        #     # now first change the conversion factor to Jy
        #     if old_unit == 'mJy':
        #         conversion_factor *= 1e-3
        #     elif old_unit == 'MJy/sr':
        #         conversion_factor *= (1e6 * pixel_size)
        #     elif old_unit == 'erg A-1 cm-2 s-1':
        #         # The conversion from erg A-1 cm-2 s-1 is well described in
        #         # https://www.physicsforums.com/threads/unit-conversion-flux-densities.742561/
        #         # se also
        #         # https://www.physicsforums.com/threads/unit-conversion-of-flux-jansky-to-erg-s-cm-a-simplified-guide.927166/
        #         # we use fv dv = fλ dλ
        #         # fλ = fv dv/dλ
        #         # and because v = c/λ...
        #         # fλ = fv*c / λ^2
        #         # thus the conversion factor is:
        #         conversion_factor = 1e23 * 1e-8 * (band_wave ** 2) / (speed_of_light * 1e2)
        #         # the speed of light is in m/s the factor 1-e2 changes it to cm/s
        #         # the factor 1e8 changes Angstrom to cm (the Angstrom was in the nominator therefore it is 1/1e-8)
        #
        #     # now convert to new unit
        #     if new_unit == 'mJy':
        #         conversion_factor *= 1e3
        #     elif new_unit == 'MJy/sr':
        #         conversion_factor *= 1e-6 / pixel_size
        #     elif new_unit == 'erg A-1 cm-2 s-1':
        #         conversion_factor *= 1e-23 * 1e8 * (speed_of_light * 1e2) / (band_wave ** 2)

        conversion_factor = helper_func.UnitTools.get_flux_unit_conv_fact(old_unit=old_unit, new_unit=new_unit,
                                                                          pixel_size=pixel_size, band_wave=band_wave)

        # change data
        getattr(self, '%s_bands_data' % instrument)['%s_data_img' % band] *= conversion_factor
        getattr(self, '%s_bands_data' % instrument)['%s_unit_img' % band] = new_unit
        if '%s_data_err' % band in getattr(self, '%s_bands_data' % instrument).keys():
            getattr(self, '%s_bands_data' % instrument)['%s_data_err' % band] *= conversion_factor

    def get_band_cutout_dict(self, ra_cutout, dec_cutout, cutout_size, include_err=False, band_list=None):
        """

        Parameters
        ----------
        ra_cutout : float
        dec_cutout : float
        cutout_size : float, tuple or list
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.
        include_err : bool
        band_list : list

        Returns
        -------
        cutout_dict : dict
        each element in dictionary is of type astropy.nddata.Cutout2D object
        """
        # geta list with all observed bands in order of wavelength
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        if not isinstance(cutout_size, list):
            cutout_size = [cutout_size] * len(band_list)

        cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')
        cutout_dict = {'cutout_pos': cutout_pos}
        cutout_dict.update({'cutout_size': cutout_size})
        cutout_dict.update({'band_list': band_list})

        for band, band_index in zip(band_list, range(len(band_list))):
            if band in helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.CoordTools.get_img_cutout(img=self.hst_bands_data['%s_data_img' % band],
                                                              wcs=self.hst_bands_data['%s_wcs_img' % band],
                                                              coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.CoordTools.get_img_cutout(img=self.hst_bands_data['%s_data_err' % band],
                                                                  wcs=self.hst_bands_data['%s_wcs_err' % band],
                                                                  coord=cutout_pos,
                                                                  cutout_size=cutout_size[band_index])})
            # if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            #     if band == helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name):
            #         cutout_dict.update({
            #             '%s_img_cutout' % band:
            #                 helper_func.CoordTools.get_img_cutout(img=self.hst_bands_data['%s_data_img' % band],
            #                                                       wcs=self.hst_bands_data['%s_wcs_img' % band],
            #                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})
            #         if include_err:
            #             cutout_dict.update({
            #                 '%s_err_cutout' % band:
            #                     helper_func.CoordTools.get_img_cutout(
            #                         img=self.hst_bands_data['%s_data_err' % band],
            #                         wcs=self.hst_bands_data['%s_wcs_err' % band],
            #                         coord=cutout_pos, cutout_size=cutout_size[band_index])})
            if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
                if band == (helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name) + '_cont_sub'):
                    cutout_dict.update({
                        '%s_img_cutout' % band:
                            helper_func.CoordTools.get_img_cutout(img=self.hst_ha_cont_sub_bands_data['%s_data_img' % band],
                                                                  wcs=self.hst_ha_cont_sub_bands_data['%s_wcs_img' % band],
                                                                  coord=cutout_pos, cutout_size=cutout_size[band_index])})
                    if include_err:
                        cutout_dict.update({
                            '%s_err_cutout' % band:
                                helper_func.CoordTools.get_img_cutout(
                                    img=self.hst_ha_cont_sub_bands_data['%s_data_err' % band],
                                    wcs=self.hst_ha_cont_sub_bands_data['%s_wcs_err' % band],
                                    coord=cutout_pos, cutout_size=cutout_size[band_index])})

            if helper_func.ObsTools.check_nircam_obs(target=self.phot_nircam_target_name, version=self.nircam_data_ver):
                if band in helper_func.ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                         version=self.nircam_data_ver):
                    cutout_dict.update({
                        '%s_img_cutout' % band:
                            helper_func.CoordTools.get_img_cutout(img=self.nircam_bands_data['%s_data_img' % band],
                                                                  wcs=self.nircam_bands_data['%s_wcs_img' % band],
                                                                  coord=cutout_pos, cutout_size=cutout_size[band_index])})
                    if include_err:
                        cutout_dict.update({
                            '%s_err_cutout' % band:
                                helper_func.CoordTools.get_img_cutout(img=self.nircam_bands_data['%s_data_err' % band],
                                                                      wcs=self.nircam_bands_data['%s_wcs_err' % band],
                                                                      coord=cutout_pos,
                                                                      cutout_size=cutout_size[band_index])})
            if helper_func.ObsTools.check_miri_obs(target=self.phot_miri_target_name, version=self.miri_data_ver):
                if band in helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                       version=self.miri_data_ver):
                    cutout_dict.update({
                        '%s_img_cutout' % band:
                            helper_func.CoordTools.get_img_cutout(img=self.miri_bands_data['%s_data_img' % band],
                                                                  wcs=self.miri_bands_data['%s_wcs_img' % band],
                                                                  coord=cutout_pos, cutout_size=cutout_size[band_index])})
                    if include_err:
                        cutout_dict.update({
                            '%s_err_cutout' % band:
                                helper_func.CoordTools.get_img_cutout(img=self.miri_bands_data['%s_data_err' % band],
                                                                      wcs=self.miri_bands_data['%s_wcs_err' % band],
                                                                      coord=cutout_pos,
                                                                      cutout_size=cutout_size[band_index])})

            if helper_func.ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name, version=self.astrosat_data_ver):
                if band in helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name,
                                                                           version=self.astrosat_data_ver):
                    cutout_dict.update({
                        '%s_img_cutout' % band:
                            helper_func.CoordTools.get_img_cutout(img=self.astrosat_bands_data['%s_data_img' % band],
                                                                  wcs=self.astrosat_bands_data['%s_wcs_img' % band],
                                                                  coord=cutout_pos, cutout_size=cutout_size[band_index])})
                    if include_err:
                        cutout_dict.update({
                            '%s_err_cutout' % band:
                                helper_func.CoordTools.get_img_cutout(img=self.astrosat_bands_data['%s_data_err' % band],
                                                                      wcs=self.astrosat_bands_data['%s_wcs_err' % band],
                                                                      coord=cutout_pos,
                                                                      cutout_size=cutout_size[band_index])})
        return cutout_dict

    def get_hst_median_exp_time(self, band):
        """
        Function to calculate the median exposure time of HST observations
        Parameters
        ----------
        band : str

        Returns
        -------
        median_exp_time : float
        """
        exp_file_name = self.get_hst_img_file_name(band=band, file_type='wht')
        if self.phot_hst_target_name == 'ngc5194': hdu_number = 'WHT'
        else: hdu_number = 0
        data, header, wcs = helper_func.FileTools.load_img(file_name=exp_file_name, hdu_number=hdu_number)
        return np.nanmedian(data[data != 0])

    def get_hst_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of HST observations

        Returns
        -------
        coverage_dict : dict
        """
        if not os.path.isfile(self.path2obs_cover_hull / ('%s_hst_obs_hull_dict.npy' % self.phot_hst_target_name)):
            return None
        with open(self.path2obs_cover_hull / ('%s_hst_obs_hull_dict.npy' % self.phot_hst_target_name), 'rb') as file_name:
            return pickle.load(file_name)

    def get_nircam_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of NIRCAM observations

        Returns
        -------
        coverage_dict : dict
        """
        if not os.path.isfile(self.path2obs_cover_hull / ('%s_nircam_obs_hull_dict_%s.npy' % (self.phot_nircam_target_name, self.nircam_data_ver))):
            return None
        with open(self.path2obs_cover_hull / ('%s_nircam_obs_hull_dict_%s.npy' % (self.phot_nircam_target_name, self.nircam_data_ver)), 'rb') as file_name:
            return pickle.load(file_name)

    def get_miri_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of MIRI observations

        Returns
        -------
        coverage_dict : dict
        """
        if not os.path.isfile(self.path2obs_cover_hull / ('%s_miri_obs_hull_dict_%s.npy' % (self.phot_miri_target_name, self.miri_data_ver))):
            return None
        with open(self.path2obs_cover_hull / ('%s_miri_obs_hull_dict_%s.npy' % (self.phot_miri_target_name, self.miri_data_ver)), 'rb') as file_name:
            return pickle.load(file_name)

    def get_astrosat_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of AstroSat observations

        Returns
        -------
        coverage_dict : dict
        """

        if not os.path.isfile(self.path2obs_cover_hull / ('%s_astrosat_obs_hull_dict_%s.npy' % (self.phot_astrosat_target_name, self.astrosat_data_ver))):
            return None
        with open(self.path2obs_cover_hull / ('%s_astrosat_obs_hull_dict_%s.npy' % (self.phot_astrosat_target_name, self.astrosat_data_ver)), 'rb') as file_name:
            return pickle.load(file_name)

    def check_coords_covered_by_band(self, telescope, ra, dec, band, max_dist_dist2hull_arcsec=2):
        """
        Function to check if coordinate points are inside band observations

        Parameters
        ----------
        telescope : str
        ra : float or ``np.ndarray``
        dec : float or ``np.ndarray``
        band : str
        max_dist_dist2hull_arcsec : float

        Returns
        -------
        coverage_dict : ``np.ndarray``
        """

        assert telescope in ['hst', 'nircam', 'miri', 'astrosat']
        band_hull_dict = getattr(self, 'get_%s_obs_coverage_hull_dict' % telescope)()
        if band_hull_dict is None:
            return False

        if isinstance(ra, float):
            ra = [ra]
            dec = [dec]
        coverage_mask = np.zeros(len(ra), dtype=bool)
        hull_data_ra = np.array([])
        hull_data_dec = np.array([])

        for hull_idx in band_hull_dict[band].keys():
            ra_hull = band_hull_dict[band][hull_idx]['ra']
            dec_hull = band_hull_dict[band][hull_idx]['dec']
            hull_data_ra = np.concatenate([hull_data_ra, ra_hull])
            hull_data_dec = np.concatenate([hull_data_dec, dec_hull])

            coverage_mask += helper_func.GeometryTools.check_points_in_polygon(x_point=ra, y_point=dec,
                                                                               x_data_hull=ra_hull,
                                                                               y_data_hull=dec_hull)
            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.cla()
            # plt.plot(hull_data_ra, hull_data_dec)
            # plt.scatter(ra, dec)
            # plt.show()

        coverage_mask *= helper_func.GeometryTools.flag_close_points2ensemble(
            x_data=ra, y_data=dec, x_data_ensemble=hull_data_ra,y_data_ensemble=hull_data_dec,
            max_dist2ensemble=max_dist_dist2hull_arcsec/3600)

        return coverage_mask

    def check_coords_covered_by_telescope(self, telescope, ra, dec, band_list=None, max_dist_dist2hull_arcsec=2):
        """
        Function to check if coordinate points are inside all observations

        Parameters
        ----------
        telescope : str
        ra : float or ``np.ndarray``
        dec : float or ``np.ndarray``
        band_list : list
        max_dist_dist2hull_arcsec : float

        Returns
        -------
        coverage_dict : ``np.ndarray``
        """

        assert telescope in ['hst', 'nircam', 'miri', 'astrosat']

        target_name = getattr(self, 'phot_%s_target_name' % telescope)

        if band_list is None:
            if telescope in ['nircam', 'miri', 'astrosat']:
                band_list = getattr(helper_func.ObsTools, 'get_%s_obs_band_list' % telescope)(target=target_name, version=getattr(self, '%s_data_ver' % telescope))
            else:
                band_list = getattr(helper_func.ObsTools, 'get_%s_obs_band_list' % telescope)(target=target_name)

        coverage_mask = np.ones(len(ra), dtype=bool)
        for band in band_list:
            coverage_mask *= self.check_coords_covered_by_band(telescope=telescope, ra=ra, dec=dec, band=band,
                                                               max_dist_dist2hull_arcsec=max_dist_dist2hull_arcsec)
        return coverage_mask

    def get_dss_img(self, img_rad_arcsec, survey='DSS', pixels_size=(500, 500)):
        # load DSS image
        print('...')
        SkyView.clear_cache()
        sv = SkyView()
        paths_dss = sv.get_images(position='NGC 628',
                                  survey='GALEX Near UV',
                                  # radius=img_rad_arcsec*u.arcsec,
                                  #      pixels=pixels_size
                                  )
        print('...')

        data_dss = paths_dss[0][0].data
        wcs_dss = WCS(paths_dss[0][0].header)
        return data_dss, wcs_dss

    def get_phangs_band_pc_scale_map_filepath(self, pc_scale, band, obs, flux_unit='mJy'):
        target_name = getattr(self, 'phot_%s_target_name' % obs)

        path2scale_map = Path(phangs_access_config.phangs_config_dict['scale_decomposition_path']) / target_name /  obs

        file_name = 'scale_map_%i_pc_%s_%s_flux_unit_%s.fits' % (pc_scale, target_name, band, flux_unit.replace('/', '_'))
        return path2scale_map / file_name

    def get_phangs_band_sig_scale_map_filepath(self, sig_scale, band, obs, flux_unit='mJy'):
        target_name = getattr(self, 'phot_%s_target_name' % obs)

        path2scale_map = Path(phangs_access_config.phangs_config_dict['scale_decomposition_path']) / target_name /  obs

        file_name = 'scale_map_%i_sig_%s_%s_flux_unit_%s.fits' % (sig_scale, target_name, band, flux_unit.replace('/', '_'))
        return path2scale_map / file_name

    def get_phangs_band_pc_scale_map(self, pc_scale, band, obs, flux_unit='mJy'):
        file_name = self.get_phangs_band_pc_scale_map_filepath(pc_scale=pc_scale, band=band, obs=obs,
                                                            flux_unit=flux_unit)
        scale_data, scale_header, scale_wcs = helper_func.FileTools.load_img(file_name=file_name)
        return scale_data, scale_header, scale_wcs

    def get_phangs_band_sig_scale_map(self, sig_scale, band, obs, flux_unit='mJy'):
        file_name = self.get_phangs_band_sig_scale_map_filepath(sig_scale=sig_scale, band=band, obs=obs,
                                                            flux_unit=flux_unit)
        scale_data, scale_header, scale_wcs = helper_func.FileTools.load_img(file_name=file_name)
        return scale_data, scale_header, scale_wcs

    def compute_apert_photometry(self, ra, dec, band, obs, flux_unit='mJy'):

        # make sure that data has been loaded
        self.load_phangs_bands(band_list=band, flux_unit=flux_unit, load_err=True,
                               load_hst=True, load_nircam=True, load_miri=True, load_astrosat=True)

        img_wcs = getattr(self, '%s_bands_data' % obs)['%s_wcs_img' % band]

        # get the cutout size (just twice the outer radius)

        print(img_wcs)
        exit()

        # cutout_dict = self.get_band_cutout_dict(ra_cutout, dec_cutout, cutout_size, include_err=False, band_list=None)
        #
        # # compute cutout
        # if obs == 'hst':
        #     img_data = self.hst_bands_data['%s_data_img' % band]
        #     '%s_data_img' % band: img_data, '%s_header_img'
        # img_wcs = self.get
        #
        #
        # phot_tools.ApertTools.compute_apert_photometry(data, data_err, wcs, ra, dec, band, obs, mask=None,
        #                                                sigma_clip_sig=3,
        #                                                sigma_clip_maxiters=5)






    def compute_2d_region_bkg(self, ra, dec, band, instrument, cutout_size, bkg_cutout_size=None,
                              bkg_img_size_factor=40, box_size_factor=2, filter_size_factor=1,
                              do_sigma_clip=True, sigma=3.0, maxiters=10, bkg_method='SExtractorBackground'):
        """
        This function creates a 2D background map for a given region of size ``cutout_size``.
        However, The parameters to calculate the background should depend on the size of the PSF!
        Thus, the size on the original cutout used for the BKG estimation will be calculated using
        ``bkg_img_size_factor``. In case The computed bkg_size is smaller than the actual ``cutout_size``
        we will adapt the actual cutout size. Nevertheless, we give the option to give a custom bkg_cutout_size.

        Parameters

        """
        # get scaling of the PSF
        fwhm_arcsec = phot_tools.PSFTools.get_obs_psf_fwhm(band=band, instrument=instrument)
        # get bkg cutout size
        if bkg_cutout_size is None:
            # bkg_cutout_size = bkg_img_size_factor * fwhm_arcsec
            if cutout_size[0] <= bkg_img_size_factor * fwhm_arcsec:
                bkg_cutout_size_x = bkg_img_size_factor * fwhm_arcsec
            else:
                bkg_cutout_size_x = cutout_size[0] + fwhm_arcsec * 5
            if cutout_size[1] <= bkg_img_size_factor * fwhm_arcsec:
                bkg_cutout_size_y = bkg_img_size_factor * fwhm_arcsec
            else:
                bkg_cutout_size_y = cutout_size[1] + fwhm_arcsec * 5
            bkg_cutout_size = (bkg_cutout_size_x, bkg_cutout_size_y)
        cutout_dict_bkg = self.get_band_cutout_dict(
            ra_cutout=ra, dec_cutout=dec, cutout_size=bkg_cutout_size, band_list=[band], include_err=False)

        cutout_stamp_bkg, cutout_stamp_bkg_rms = phot_tools.BKGTools.get_scaled_bkg(
            ra=ra, dec=dec, cutout_size=cutout_size, bkg_cutout=cutout_dict_bkg['%s_img_cutout' % band].data,
            bkg_wcs=cutout_dict_bkg['%s_img_cutout' % band].wcs, scale_size_arcsec=fwhm_arcsec,
            box_size_factor=box_size_factor, filter_size_factor=filter_size_factor,
            do_sigma_clip=do_sigma_clip, sigma=sigma, maxiters=maxiters, bkg_method=bkg_method)

        return cutout_stamp_bkg, cutout_stamp_bkg_rms

    def get_region_topo(self,
                        ra, dec, band, instrument, cutout_size,
                        # keywords for source statistics
                        src_stats_rad_factor=3,
                        # keywords for background estimation
                        bkg_cutout_size=None, bkg_img_size_factor=40, bkg_box_size_factor=2,
                        bkg_filter_size_factor=1, bkg_do_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
                        bkg_method='SExtractorBackground'):
        # get the source cutout
        # get source cutout
        cutout_dict_src = self.get_band_cutout_dict(
            ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band], include_err=True)
        # get the bkg cutout
        cutout_stamp_bkg, cutout_stamp_bkg_rms = self.compute_2d_region_bkg(
            ra=ra, dec=dec, band=band, instrument=instrument, cutout_size=cutout_size, bkg_cutout_size=bkg_cutout_size,
            bkg_img_size_factor=bkg_img_size_factor, box_size_factor=bkg_box_size_factor,
            filter_size_factor=bkg_filter_size_factor, do_sigma_clip=bkg_do_sigma_clip, sigma=bkg_sigma, maxiters=bkg_maxiters,
            bkg_method=bkg_method)

        # get a mask of bad pixels!
        # depending on the pipeline bad values are either nan values or they are infinite values
        mask_bad_pixels = (np.isnan(cutout_dict_src['%s_img_cutout' % band].data) +
                           np.isinf(cutout_dict_src['%s_img_cutout' % band].data) +
                           (cutout_dict_src['%s_img_cutout' % band].data == 0))

        # get band STD and FWHM in order to perform source detection
        # we need an approximation of the size of one PSF
        psf_std = phot_tools.PSFTools.get_obs_psf_std(band=band, instrument=instrument)
        psf_std_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=psf_std, wcs=cutout_dict_src['%s_img_cutout' % band].wcs).value
        psf_fwhm = phot_tools.PSFTools.get_obs_psf_fwhm(band=band, instrument=instrument)
        psf_fwhm_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=psf_fwhm, wcs=cutout_dict_src['%s_img_cutout' % band].wcs).value

        x_pos, y_pos = cutout_dict_src['%s_img_cutout' % band].wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg))

        # get photometric statistics of the central position
        img_central_stats = phot_tools.ApertTools.get_apert_stats(
            data=cutout_dict_src['%s_img_cutout' % band].data, data_err=cutout_dict_src['%s_err_cutout' % band].data,
            x_pos=x_pos, y_pos=y_pos, aperture_rad_pix=src_stats_rad_factor * psf_std_pix)
        bkg_central_stats = phot_tools.ApertTools.get_apert_stats(
            data=cutout_stamp_bkg.data, data_err=cutout_dict_src['%s_err_cutout' % band].data,
            x_pos=x_pos, y_pos=y_pos, aperture_rad_pix=src_stats_rad_factor * psf_std_pix)

        region_topo_dict = {'ra': ra, 'dec': dec, 'x_pos': x_pos, 'y_pos': y_pos,
                            'img': cutout_dict_src['%s_img_cutout' % band].data,
                            'img_err': cutout_dict_src['%s_err_cutout' % band].data,
                            'bkg': cutout_stamp_bkg.data,
                            'bkg_rms': cutout_stamp_bkg_rms.data,
                            'wcs': cutout_dict_src['%s_img_cutout' % band].wcs,
                            'mask_bad_pixels': mask_bad_pixels,
                            'img_central_stats': img_central_stats,
                            'bkg_central_stats': bkg_central_stats,
                            'psf_std': psf_std, 'psf_std_pix': psf_std_pix,
                            'psf_fwhm': psf_fwhm, 'psf_fwhm_pix': psf_fwhm_pix,
                            }

        return region_topo_dict

    def compute_morph_phot(self, ra, dec, band, instrument, cutout_size,
                        src_stats_rad_factor=3, bkg_img_size_factor=60, bkg_box_size_factor=2,
                        bkg_filter_size_factor=1, bkg_do_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
                        bkg_method='SExtractorBackground',
                           profile_n_slits=12, max_rad_profile_fact=3,
                           morph_fit_upper_sig_fact=10,
                           morph_fit_central_rad_fact=3, morph_fit_model_pos_rad_accept_fact=1):

        region_topo_dict = self.get_region_topo(
            ra=ra, dec=dec, band=band, instrument=instrument, cutout_size=cutout_size,
            src_stats_rad_factor=src_stats_rad_factor, bkg_img_size_factor=bkg_img_size_factor,
            bkg_box_size_factor=bkg_box_size_factor, bkg_filter_size_factor=bkg_filter_size_factor,
            bkg_do_sigma_clip=bkg_do_sigma_clip, bkg_sigma=bkg_sigma, bkg_maxiters=bkg_maxiters, bkg_method=bkg_method)

        profile_dict_src_bkg_sub = phot_tools.ProfileTools.get_rad_profile_dict(
            img=region_topo_dict['img'] - region_topo_dict['bkg'], wcs=region_topo_dict['wcs'],
            ra=ra, dec=dec, n_slits=profile_n_slits,
            max_rad_arcsec=region_topo_dict['psf_fwhm'] * max_rad_profile_fact,
            img_err=region_topo_dict['img_err'],
            img_mask=region_topo_dict['mask_bad_pixels'])
        pos_pix = region_topo_dict['wcs'].world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg))

        morph_dict = phot_tools.ProfileTools.measure_morph_photometry(
            topo_dict=region_topo_dict, band=band, instrument=instrument,
            x_center=pos_pix[0], ycenter=pos_pix[1],
            rad_profile_dict=profile_dict_src_bkg_sub['slit_profile_dict'],
            std_pix=region_topo_dict['psf_std_pix'], upper_sig_fact=morph_fit_upper_sig_fact,
            central_rad_fact=morph_fit_central_rad_fact, model_pos_rad_accept_fact=morph_fit_model_pos_rad_accept_fact)

        # print(morph_dict)

        # import matplotlib.pyplot as plt
        # plt.close()
        # plt.imshow(region_topo_dict['img'])
        # plt.show()

        return morph_dict



    def compute_ci(self, ra, dec, band, telescope, rad_1_pix=1, rad_2_pix=3
                   ):

        img = getattr(self,'%s_bands_data' % telescope)['%s_data_img' % band]
        wcs = getattr(self,'%s_bands_data' % telescope)['%s_wcs_img' % band]

        rad_1_arcsec = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=rad_1_pix, wcs=wcs)
        rad_2_arcsec = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=rad_2_pix, wcs=wcs)

        return phot_tools.ApertTools.compute_annulus_ci(img=img,
                                                 img_err=None, wcs=wcs,
            ra=ra, dec=dec, rad_1_arcsec=rad_1_arcsec, rad_2_arcsec=rad_2_arcsec)





    def compute_ci_old(self, ra, dec, band, instrument, cutout_size,
                        src_stats_rad_factor=3, bkg_img_size_factor=60, bkg_box_size_factor=2,
                        bkg_filter_size_factor=1, bkg_do_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
                        bkg_method='SExtractorBackground'):

        # get psf
        psf_std = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        rad_1_arcsec = psf_std['ee_25_arcsec']
        rad_2_arcsec = psf_std['ee_80_arcsec']

        region_topo_dict = self.get_region_topo(
            ra=ra, dec=dec, band=band, instrument=instrument, cutout_size=cutout_size,
            src_stats_rad_factor=src_stats_rad_factor, bkg_img_size_factor=bkg_img_size_factor,
            bkg_box_size_factor=bkg_box_size_factor, bkg_filter_size_factor=bkg_filter_size_factor,
            bkg_do_sigma_clip=bkg_do_sigma_clip, bkg_sigma=bkg_sigma, bkg_maxiters=bkg_maxiters, bkg_method=bkg_method)

        return phot_tools.ApertTools.compute_annulus_ci(img=region_topo_dict['img'] - region_topo_dict['bkg'],
                                                 img_err=region_topo_dict['img'], wcs=region_topo_dict['wcs'],
            ra=ra, dec=dec, rad_1_arcsec=rad_1_arcsec, rad_2_arcsec=rad_2_arcsec)




    #
    # def src_detect_in_cutout(self,
    #                          ra, dec, band, instrument, cutout_size, src_threshold_detect_factor=3,
    #                          src_fwhm_detect_factor=1,
    #                          # keywords for source statistics
    #                          src_stats_rad_factor=3,
    #                          # keywords for background estimation
    #                          bkg_cutout_size=None, bkg_img_size_factor=40, bkg_box_size_factor=2,
    #                          bkg_filter_size_factor=1, bkg_do_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
    #                          bkg_method='SExtractorBackground'):
    #
    #     # get the source cutout
    #     # get source cutout
    #     cutout_dict_src = self.get_band_cutout_dict(
    #         ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band], include_err=True)
    #     # get the bkg cutout
    #     cutout_stamp_bkg, cutout_stamp_bkg_rms = self.compute_2d_region_bkg(
    #         ra=ra, dec=dec, band=band, instrument=instrument, cutout_size=cutout_size, bkg_cutout_size=bkg_cutout_size,
    #         bkg_img_size_factor=bkg_img_size_factor, box_size_factor=bkg_box_size_factor,
    #         filter_size_factor=bkg_filter_size_factor, do_sigma_clip=bkg_do_sigma_clip, sigma=bkg_sigma, maxiters=bkg_maxiters,
    #         bkg_method=bkg_method)
    #
    #     # get band STD and FWHM in order to perform source detection
    #     # we need an approximation of the size of one PSF
    #     psf_std = phot_tools.PSFTools.get_obs_psf_std(band=band, instrument=instrument)
    #     psf_std_pix = helper_func.CoordTools.transform_world2pix_scale(
    #         length_in_arcsec=psf_std, wcs=cutout_dict_src['%s_img_cutout' % band].wcs).value
    #     psf_fwhm = phot_tools.PSFTools.get_obs_psf_fwhm(band=band, instrument=instrument)
    #     psf_fwhm_pix = helper_func.CoordTools.transform_world2pix_scale(
    #         length_in_arcsec=psf_fwhm, wcs=cutout_dict_src['%s_img_cutout' % band].wcs).value
    #
    #     x_pos, y_pos = cutout_dict_src['%s_img_cutout' % band].wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg))
    #
    #     # get photometric statistics of the central position
    #     img_central_stats = phot_tools.ApertTools.get_sky_apert_stats(
    #         data=cutout_dict_src['%s_img_cutout' % band].data, data_err=cutout_dict_src['%s_err_cutout' % band].data,
    #         x_pos=x_pos, y_pos=y_pos, aperture_rad=src_stats_rad_factor * psf_std_pix)
    #     bkg_central_stats = phot_tools.ApertTools.get_sky_apert_stats(
    #         data=cutout_stamp_bkg.data, data_err=cutout_dict_src['%s_err_cutout' % band].data,
    #         x_pos=x_pos, y_pos=y_pos, aperture_rad=src_stats_rad_factor * psf_std_pix)
    #
    #     # perform source detection
    #     dao_detection = phot_tools.SrcTools.detect_star_like_src_in_band_cutout(
    #         data=cutout_dict_src['%s_img_cutout' % band].data - cutout_stamp_bkg.data,
    #         detection_threshold=src_threshold_detect_factor * np.nanmedian(cutout_stamp_bkg_rms.data),
    #         psf_fwhm_pix=src_fwhm_detect_factor * psf_fwhm_pix)
    #     # get detected sources in
    #     if dao_detection is None:
    #         x_src = []
    #         y_src = []
    #         ra_src = []
    #         dec_src = []
    #     else:
    #         x_src = list(dao_detection['xcentroid'])
    #         y_src = list(dao_detection['ycentroid'])
    #         positions_world = cutout_dict_src['%s_img_cutout' % band].wcs.pixel_to_world(
    #             dao_detection['xcentroid'], dao_detection['ycentroid'])
    #         ra_src = list(positions_world.ra.deg)
    #         dec_src = list(positions_world.dec.deg)
    #
    #     src_dict = {'img': cutout_dict_src['%s_img_cutout' % band].data,
    #                 'img_err': cutout_dict_src['%s_err_cutout' % band].data,
    #                 'bkg': cutout_stamp_bkg.data,
    #                 'bkg_rms': cutout_stamp_bkg_rms.data,
    #                 'wcs': cutout_dict_src['%s_img_cutout' % band].wcs,
    #                 'img_central_stats': img_central_stats,
    #                 'bkg_central_stats': bkg_central_stats,
    #                 'psf_fwhm': psf_fwhm, 'psf_fwhm_pix': psf_fwhm_pix,
    #                 'x_src': x_src, 'y_src': y_src, 'ra_src': ra_src, 'dec_src': dec_src}
    #
    #     return src_dict
    #
    # def recenter_src(self, ra, dec, band, instrument, cutout_size, src_threshold_detect_factor=3,
    #                  src_stats_rad_factor=1, bkg_cutout_size=None, bkg_img_size_factor=40, bkg_box_size_factor=2,
    #                  bkg_filter_size_factor=1, bkg_do_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
    #                  bkg_method='SExtractorBackground'):
    #
    #     # get src dict
    #     src_dict = self.src_detect_in_cutout(
    #         ra=ra, dec=dec, band=band, instrument=instrument, cutout_size=cutout_size,
    #         src_threshold_detect_factor=src_threshold_detect_factor,
    #         src_stats_rad_factor=src_stats_rad_factor,
    #         bkg_cutout_size=bkg_cutout_size, bkg_img_size_factor=bkg_img_size_factor,
    #         bkg_box_size_factor=bkg_box_size_factor, bkg_filter_size_factor=bkg_filter_size_factor,
    #         bkg_do_sigma_clip=bkg_do_sigma_clip, bkg_sigma=bkg_sigma, bkg_maxiters=bkg_maxiters, bkg_method=bkg_method)
    #
    #     # check whether there is a source inside the FWHM of the original source
    #     init_pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    #     init_pos_pix = src_dict['wcs'].world_to_pixel(init_pos)
    #     if src_dict['ra_src']:
    #         pos_src = SkyCoord(ra=src_dict['ra_src']*u.deg, dec=src_dict['dec_src']*u.deg)
    #         separation = pos_src.separation(init_pos)
    #         mask_src_inside_psf_fwhm = separation < src_dict['psf_fwhm'] * u.arcsec
    #         if sum(mask_src_inside_psf_fwhm) == 0:
    #             ra_src_recenter, dec_src_recenter = ra, dec
    #             x_src_recenter, y_src_recenter = init_pos_pix
    #         else:
    #             mask_closest_src = separation == np.min(separation)
    #             ra_src_recenter = float(pos_src[mask_closest_src].ra.deg)
    #             dec_src_recenter = float(pos_src[mask_closest_src].dec.deg)
    #             x_src_recenter = float(src_dict['wcs'].world_to_pixel(pos_src[mask_closest_src])[0])
    #             y_src_recenter = float(src_dict['wcs'].world_to_pixel(pos_src[mask_closest_src])[1])
    #
    #     else:
    #         ra_src_recenter, dec_src_recenter = ra, dec
    #         x_src_recenter, y_src_recenter = init_pos_pix
    #
    #     src_dict.update({'ra_src_recenter': ra_src_recenter, 'dec_src_recenter': dec_src_recenter,
    #                      'x_src_recenter': x_src_recenter, 'y_src_recenter': y_src_recenter})
    #
    #     return src_dict


    """
    old, soon to be discarded functions 
    """
    def compute_hst_ew_aprt_corr(self, ra, dec, cutout_size, left_band, right_band, hst_ha_band):
        band_list = [left_band] + [right_band] + [hst_ha_band]

        # load data in case it is not yet loaded
        self.load_phangs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)

        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                                      band_list=band_list, include_err=True)

        flux_dict_left_band_appr_corr = PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                              data=cutout_dict_stamp[
                                                                                  '%s_img_cutout' % left_band].data,
                                                                              err=cutout_dict_stamp[
                                                                                  '%s_err_cutout' % left_band].data,
                                                                              wcs=cutout_dict_stamp[
                                                                                  '%s_img_cutout' % left_band].wcs,
                                                                              obs='hst', band=left_band)
        flux_dict_right_band_appr_corr = PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                               data=cutout_dict_stamp[
                                                                                   '%s_img_cutout' % right_band].data,
                                                                               err=cutout_dict_stamp[
                                                                                   '%s_err_cutout' % right_band].data,
                                                                               wcs=cutout_dict_stamp[
                                                                                   '%s_img_cutout' % right_band].wcs,
                                                                               obs='hst', band=right_band)
        flux_dict_ha_appr_corr = PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                       data=cutout_dict_stamp[
                                                                           '%s_img_cutout' % hst_ha_band].data,
                                                                       err=cutout_dict_stamp[
                                                                           '%s_err_cutout' % hst_ha_band].data,
                                                                       wcs=cutout_dict_stamp[
                                                                           '%s_img_cutout' % hst_ha_band].wcs,
                                                                       obs='hst_ha', band=hst_ha_band)
        # compute EW
        return PhotTools.compute_hst_photo_ew(
            target=self.phot_hst_target_name, left_band=left_band, right_band=right_band, narrow_band=hst_ha_band,
            flux_left_band=flux_dict_left_band_appr_corr['flux'],
            flux_right_band=flux_dict_right_band_appr_corr['flux'],
            flux_narrow_band=flux_dict_ha_appr_corr['flux'],
            flux_err_left_band=flux_dict_left_band_appr_corr['flux_err'],
            flux_err_right_band=flux_dict_right_band_appr_corr['flux_err'],
            flux_err_narrow_band=flux_dict_ha_appr_corr['flux_err'])

    def compute_hst_ew_at_rad(self, ra, dec, cutout_size, apert_rad, annulus_rad_in, annulus_rad_out,
                              left_band, right_band, hst_ha_band):
        band_list = [left_band] + [right_band] + [hst_ha_band]

        # load data in case it is not yet loaded
        self.load_phangs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)

        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                                      band_list=band_list, include_err=True)
        # compute flux
        flux_dict_left_band = PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                            data=cutout_dict_stamp['%s_img_cutout' % left_band].data,
                                                            err=cutout_dict_stamp['%s_err_cutout' % left_band].data,
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs,
                                                            obs='hst', band=left_band, aperture_rad=apert_rad,
                                                            annulus_rad_in=annulus_rad_in,
                                                            annulus_rad_out=annulus_rad_out)
        flux_dict_right_band = PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                             data=cutout_dict_stamp['%s_img_cutout' % right_band].data,
                                                             err=cutout_dict_stamp['%s_err_cutout' % right_band].data,
                                                             wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs,
                                                             obs='hst', band=right_band,
                                                             aperture_rad=apert_rad,
                                                             annulus_rad_in=annulus_rad_in,
                                                             annulus_rad_out=annulus_rad_out)
        flux_dict_ha = PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                     data=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                                                     err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data,
                                                     wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                                                     obs='hst_ha', band=hst_ha_band, aperture_rad=apert_rad,
                                                     annulus_rad_in=annulus_rad_in,
                                                     annulus_rad_out=annulus_rad_out)

        return PhotTools.compute_hst_photo_ew(
            target=self.phot_hst_target_name, left_band=left_band, right_band=right_band, narrow_band=hst_ha_band,
            flux_left_band=flux_dict_left_band['flux'],
            flux_right_band=flux_dict_right_band['flux'],
            flux_narrow_band=flux_dict_ha['flux'],
            flux_err_left_band=flux_dict_left_band['flux_err'],
            flux_err_right_band=flux_dict_right_band['flux_err'],
            flux_err_narrow_band=flux_dict_ha['flux_err'])


