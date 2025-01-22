"""
Gathers all functions to estimate photometry
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from photutils import aperture_photometry
from photutils import CircularAperture, CircularAnnulus
from photutils import detect_sources, detect_threshold
from photutils.profiles import RadialProfile, CurveOfGrowth
from photutils.aperture import SkyCircularAperture, SkyCircularAnnulus, CircularAnnulus, CircularAperture
from photutils.aperture import ApertureStats
from photutils.aperture import aperture_photometry
from photutils import background
from photutils.detection import DAOStarFinder

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import constants as const
speed_of_light_kmps = const.c.to('km/s').value
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats

# owen packages
from phangs_data_access import phys_params, helper_func, phangs_info
from phangs_data_access.dust_tools import DustTools


class PSFTools:
    """
    class to access PSF information of different telescopes
    Note that this is work in progress and should so far not be used as a reference.
    Especially for JWST and HST the output PSF will be either a mean PSF or a psf at a specific detector position.
    """

    @staticmethod
    def get_closest_available_hst_psf_filter(band, instrument):
        if instrument == 'acs':
            available_band_list = phangs_info.acs_wfc_psf_band_list
            wave_list = []
            for available_band in available_band_list:
                wave_list.append(helper_func.ObsTools.get_hst_band_wave(available_band, instrument='acs',
                                                                        wave_estimator='mean_wave', unit='mu'))
            wave = helper_func.ObsTools.get_hst_band_wave(band, instrument='acs', wave_estimator='mean_wave', unit='mu')
        elif instrument == 'uvis':
            available_band_list = phangs_info.wfc3_uv_psf_band_list
            wave_list = []
            for available_band in available_band_list:
                wave_list.append(helper_func.ObsTools.get_hst_band_wave(available_band, instrument='uvis',
                                                                        wave_estimator='mean_wave', unit='mu'))
            wave = helper_func.ObsTools.get_hst_band_wave(band, instrument='uvis', wave_estimator='mean_wave',
                                                          unit='mu')
        elif instrument == 'ir':
            available_band_list = phangs_info.wfc3_ir_psf_band_list
            wave_list = []
            for available_band in available_band_list:
                wave_list.append(helper_func.ObsTools.get_hst_band_wave(available_band, instrument='ir',
                                                                        wave_estimator='mean_wave', unit='mu'))
            wave = helper_func.ObsTools.get_hst_band_wave(band, instrument='ir', wave_estimator='mean_wave', unit='mu')
        else:
            raise KeyError('instrument not understood')
        if band in available_band_list:
            return band
        else:
            # get the closest band:
            min_diff = np.min(np.abs(np.array(wave_list) - wave))
            idx_closest_wave = np.where((np.abs(np.array(wave_list) - wave)) == min_diff)[0][0]
            return available_band_list[idx_closest_wave]

    @staticmethod
    def load_hst_psf_dict(band, instrument):
        """
        Parameters
        ----------
        band : str
            HST band
        instrument : str
            must be acs, uvis or ir

        Returns
        -------
        psf_dict : dict

        """
        # get psf dict path
        path2psf_dict = Path(__file__).parent.parent.resolve() / 'meta_data' / 'psf_data' / 'data_output'
        if instrument == 'acs':
            instrument_str = 'acs_wfc'
        elif instrument == 'uvis':
            instrument_str = 'wfc3_uv'
        elif instrument == 'ir':
            instrument_str = 'wfc3_ir'
        else:
            raise KeyError('instrument not understood')

        psf_dict_filename = 'hst_%s_psf_dict.npy' % instrument_str
        # now there is not an estimated PSF for every filter at the moment hence we use the closest filter available
        with open(path2psf_dict / psf_dict_filename, 'rb') as file_name:
            psf_dict = pickle.load(file_name)
        # print(psf_dict)
        return psf_dict[PSFTools.get_closest_available_hst_psf_filter(band=band, instrument=instrument)]

    @staticmethod
    def get_hst_psf_rad_profile(band, instrument):
        """
        Parameters
        ----------
        band : str
            HST band
        instrument : str
            must be acs, uvis or ir

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_hst_psf_dict(band=band, instrument=instrument)
        return psf_dict['radius_arcsec'], psf_dict['psf_profile']

    @staticmethod
    def get_hst_psf_fwhm(band, instrument):
        """
        Parameters
        ----------
        band : str
            HST band
        instrument : str
            must be acs, uvis or ir

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_hst_psf_dict(band=band, instrument=instrument)
        return psf_dict['gaussian_fwhm']

    @staticmethod
    def get_hst_psf_gauss_approx(band, instrument, rad_arcsec, amp=1):
        """
        Parameters
        ----------
        band : str
            HST band
        instrument : str
            must be acs, uvis or ir

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_hst_psf_dict(band=band, instrument=instrument)
        mu = psf_dict['gaussian_mean']
        sig = psf_dict['gaussian_std']
        return amp * np.exp(-(rad_arcsec - mu) ** 2 / (2 * sig ** 2))

    @staticmethod
    # get psf dict path
    def load_jwst_psf_dict(band, instrument):
        """
        Parameters
        ----------
        band : str
            HST band
        instrument : str
            must be nircam or miri

        Returns
        -------
        psf_dict : dict

        """
        # get psf dict path
        assert(instrument in ['nircam', 'miri'])

        path2psf_dict = Path(__file__).parent.parent.resolve() / 'meta_data' / 'psf_data' / 'data_output'

        psf_dict_filename = '%s_psf_dict.npy' % instrument
        # now there is not an estimated PSF for every filter at the moment hence we use the closest filter available
        with open(path2psf_dict / psf_dict_filename, 'rb') as file_name:
            psf_dict = pickle.load(file_name)
        # print(psf_dict)
        return psf_dict[band]

    @staticmethod
    def get_jwst_psf_rad_profile(band, instrument):
        """
        Parameters
        ----------
        band : str
            NIRCAM or MIRI band
        instrument : str
            must nircam or miri

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_jwst_psf_dict(band=band, instrument=instrument)
        return psf_dict['radius_arcsec'], psf_dict['psf_profile']

    @staticmethod
    def get_jwst_psf_fwhm(band, instrument):
        """
        Parameters
        ----------
        band : str
            NIRCAM or MIRI band
        instrument : str
            must nircam or miri

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_jwst_psf_dict(band=band, instrument=instrument)
        return psf_dict['gaussian_fwhm']

    @staticmethod
    def get_jwst_psf_gauss_approx(band, instrument, rad_arcsec, amp=1):
        """
        Parameters
        ----------
        band : str
            NIRCAM or MIRI band
        instrument : str
            must nircam or miri

        Returns
        -------
        rad, profile : array-like

        """
        psf_dict = PSFTools.load_jwst_psf_dict(band=band, instrument=instrument)
        mu = psf_dict['gaussian_mean']
        sig = psf_dict['gaussian_std']
        return amp * np.exp(-(rad_arcsec - mu) ** 2 / (2 * sig ** 2))

    @staticmethod
    def load_obs_psf_dict(band, instrument):
        """
        Parameters
        ----------
        band : str
        instrument : str
            must be acs, uvis, ir, nircam or miri

        Returns
        -------
        rad, profile : array-like

        """
        assert(instrument in ['acs', 'uvis', 'ir', 'nircam', 'miri'])
        if instrument in ['acs', 'uvis', 'ir']:
            return PSFTools.load_hst_psf_dict(band=band, instrument=instrument)
        elif instrument in ['nircam', 'miri']:
            return PSFTools.load_jwst_psf_dict(band=band, instrument=instrument)

    @staticmethod
    def get_obs_psf_rad_profile(band, instrument):
        psf_dict = PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        return psf_dict['radius_arcsec'], psf_dict['psf_profile']

    @staticmethod
    def get_obs_psf_fwhm(band, instrument):
        psf_dict = PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        return psf_dict['gaussian_fwhm']

    @staticmethod
    def get_obs_psf_std(band, instrument):
        psf_dict = PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        return psf_dict['gaussian_std']

    @staticmethod
    def get_obs_psf_gauss_approx(band, instrument, rad_arcsec, amp=1):
        psf_dict = PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        mu = psf_dict['gaussian_mean']
        sig = psf_dict['gaussian_std']
        return amp * np.exp(-(rad_arcsec - mu) ** 2 / (2 * sig ** 2))


class ProfileTools:
    """
    class to access PSF information of different telescopes
    Note that this is work in progress and should so far not be used as a reference.
    Especially for JWST and HST the output PSF will be either a mean PSF or a psf at a specific detector position.
    """

    @staticmethod
    def get_rad_profile(data, x_pos, y_pos, max_rad, err=None, mask=None, pix_scale=None, method='exact'):
        # now in order to not create a plateau for the central pixel or avoid step shapes we will use steps with half
        # a pixel size:
        edge_radii = np.arange(int(max_rad))
        rp = RadialProfile(data, (x_pos, y_pos), edge_radii, error=err, mask=mask, method=method)

        # get values and connected to the pixel scale
        rad = rp.radius
        gaussian_fwhm = rp.gaussian_fwhm
        gaussian_mean = rp.gaussian_fit.mean.value
        gaussian_std = rp.gaussian_fit.stddev.value
        # if there is a pixel scale given we can rescale the values
        if pix_scale is not None:
            rad *= pix_scale
            gaussian_fwhm *= pix_scale
            gaussian_mean *= pix_scale
            gaussian_std *= pix_scale

        # create return dict
        rad_profile_dict = {
            'rad': rad,
            'profile' : rp.profile,
            'profile_err': rp.profile_error,
            'gaussian_profile': rp.gaussian_profile,
            'gaussian_fwhm': gaussian_fwhm,
            'gaussian_amp': rp.gaussian_fit.amplitude.value,
            'gaussian_mean': gaussian_mean,
            'gaussian_std': gaussian_std}

        return rad_profile_dict

    @staticmethod
    def get_rad_profile_from_img(img, wcs, ra, dec, max_rad_arcsec, img_err=None, img_mask=None, norm_profile=True, method='exact'):
        # get central pixels
        central_pos = wcs.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
        max_rad_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=max_rad_arcsec, wcs=wcs, dim=0)
        # get pixel_scale
        pixel_scale = wcs.proj_plane_pixel_scales()[0].value * 3600

        rad_profile_stats = ProfileTools.get_rad_profile(data=img, x_pos=central_pos[0], y_pos=central_pos[1],
                                                         max_rad=max_rad_pix, err=img_err, mask=img_mask,
                                                         pix_scale=pixel_scale, method=method)

        if norm_profile:
            rad_profile_stats['profile_err'] /= np.nanmax(rad_profile_stats['profile'])
            rad_profile_stats['profile'] /= np.nanmax(rad_profile_stats['profile'])
        return rad_profile_stats['rad'], rad_profile_stats['profile'], rad_profile_stats['profile_err']

    @staticmethod
    def get_src_ee(data, x_pos, y_pos, max_rad, ee_values, err=None, pix_scale=None, method='exact'):
        # make sure that the zero point is not included since there is no definition for the COG
        edge_radii = np.arange(int(max_rad))[1:]
        cog = CurveOfGrowth(data, xycen=(x_pos, y_pos), radii=edge_radii, error=err, mask=None, method=method)
        cog.normalize()
        ee_values = cog.calc_radius_at_ee(ee=ee_values)
        if pix_scale is not None:
            ee_values *= pix_scale

        return ee_values

    @staticmethod
    def get_axis_profile(data, x_pos, y_pos, angle=0, err=None, mask=None):

        mask_pixels_in_slit = helper_func.GeometryTools.select_img_pix_along_line(data=data, x_pos=x_pos, y_pos=y_pos,
                                                                                  angle=angle)

        x_pixels = np.arange(data.shape[0])
        y_pixels = np.arange(data.shape[1])
        x_mesh, y_mesh = np.meshgrid(x_pixels, y_pixels)

        radial_map = np.sqrt((x_mesh - x_pos) ** 2 + (y_mesh - y_pos) ** 2)

        # swap the radial map
        if angle == 90:
            radial_map[y_mesh < y_pos] *= -1
        else:
            radial_map[x_mesh < x_pos] *= -1

        if mask is None:
            mask = np.zeros(data.shape, dtype=bool)

        profile_mask = mask[mask_pixels_in_slit]

        profile_data = data[mask_pixels_in_slit]
        radius_data = radial_map[mask_pixels_in_slit]
        # sort for radius
        sort = np.argsort(radius_data)
        profile_data = profile_data[sort]
        radius_data = radius_data[sort]

        if err is not None:
            profile_err = err[mask_pixels_in_slit]
            profile_err = profile_err[sort]
        else:
            profile_err = np.zeros(len(profile_data)) * np.nan

        return radius_data, profile_data, profile_err, profile_mask

    @staticmethod
    def compute_axis_profiles(data, x_pos, y_pos, n_slits=6, err=None, mask=None):

        list_angles = np.linspace(0, 180, n_slits + 1)[:-1]
        list_angle_idx = np.arange(len(list_angles))
        profile_dict = {'list_angles': list_angles, 'list_angle_idx': list_angle_idx}
        for idx, angle in zip(list_angle_idx, list_angles):
            radius_data, profile_data, profile_err, profile_mask = ProfileTools.get_axis_profile(
                data=data, x_pos=x_pos, y_pos=y_pos, angle=angle, err=err, mask=mask)
            profile_dict.update({str(idx): {'profile_data': profile_data, 'profile_err': profile_err,
                                            'radius_data': radius_data, 'profile_mask': profile_mask}})
        return profile_dict

    @staticmethod
    def compute_axis_profiles_from_img(img, wcs, ra, dec, max_rad_arcsec, n_slits=6, err=None, mask=None):
        # get central pixels
        central_pos = wcs.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
        # max_rad_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=max_rad_arcsec, wcs=wcs, dim=0)
        # get pixel_scale
        pixel_scale_arcsec = wcs.proj_plane_pixel_scales()[0].value * 3600

        profile_dict = ProfileTools.compute_axis_profiles(data=img, x_pos=central_pos[0], y_pos=central_pos[1],
                                                          n_slits=n_slits, err=err, mask=mask)

        for idx in profile_dict['list_angle_idx']:
            # rescale radius and select only radii within radius of interest
            # scale to pixels
            rad_dict = profile_dict[str(idx)]
            radius_data = rad_dict['radius_data'] * pixel_scale_arcsec
            # only use the data within the selected radius
            profile_data = rad_dict['profile_data'][np.abs(radius_data) < max_rad_arcsec]
            profile_err = rad_dict['profile_err'][np.abs(radius_data) < max_rad_arcsec]
            profile_mask = rad_dict['profile_mask'][np.abs(radius_data) < max_rad_arcsec]
            radius_data = rad_dict['radius_data'][np.abs(radius_data) < max_rad_arcsec]
            # now update the
            profile_dict.update({str(idx): {'profile_data': profile_data, 'profile_err': profile_err,
                                            'radius_data': radius_data, 'profile_mask': profile_mask}})

        return profile_dict

    @staticmethod
    def get_rad_profile_dict(img, wcs, ra, dec, n_slits, max_rad_arcsec, img_err=None, img_mask=None):
        # load cutout stamps

        # what is the needed background estimation for one source?
        # get first the radial profile:
        rad, profile, profile_err = ProfileTools.get_rad_profile_from_img(
            img=img,
            wcs=wcs,
            ra=ra, dec=dec,
            max_rad_arcsec=max_rad_arcsec,
            img_err=img_err,
            img_mask=img_mask,
            norm_profile=True)

        # get profiles along a slit
        slit_profile_dict = ProfileTools.compute_axis_profiles_from_img(
            img=img, wcs=wcs, ra=ra, dec=dec, max_rad_arcsec=max_rad_arcsec, n_slits=n_slits, err=img_err)

        return {'rad': rad, 'profile': profile, 'profile_err': profile_err, 'slit_profile_dict': slit_profile_dict}

    @staticmethod
    def measure_morph_photometry(rad_profile_dict, std_pix, upper_sig_fact=10, central_rad_fact=3, model_pos_rad_accept_fact=1):

        radius_of_interes = float(std_pix * central_rad_fact)

        # central_apert_stats_source = ApertTools.get_circ_apert_stats(
        #     data=data, data_err=data_err, x_pos=x_pos, y_pos=y_pos, aperture_rad=radius_of_interes)

        amp_list = []
        mu_list = []
        sig_list = []
        amp_err_list = []
        mu_err_list = []
        sig_err_list = []
        max_amp_value = 0

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=len(rad_profile_dict['list_angle_idx']) +1)


        for idx in rad_profile_dict['list_angle_idx']:
            print(rad_profile_dict[str(idx)]['profile_mask'])
            import matplotlib.pyplot as plt
            plt.plot(rad_profile_dict[str(idx)]['radius_data'], rad_profile_dict[str(idx)]['profile_data'])
            plt.show()
            mask_pixels_to_fit = ((rad_profile_dict[str(idx)]['radius_data'] > radius_of_interes * -1) &
                           (rad_profile_dict[str(idx)]['radius_data'] < radius_of_interes) & np.invert(rad_profile_dict[str(idx)]['profile_mask']))


            min_value_in_center = np.min(rad_profile_dict[str(idx)]['profile_data'][mask_pixels_to_fit])
            max_value_in_center = np.max(rad_profile_dict[str(idx)]['profile_data'][mask_pixels_to_fit])

            lower_amp = min_value_in_center
            upper_amp = max_value_in_center + np.abs(max_value_in_center * 2)

            # update the maximal amplitude value
            if max_value_in_center > max_amp_value:
                max_amp_value = max_value_in_center

            # ax[idx].plot(rad_profile_dict[str(idx)]['radius_data'],
            #              rad_profile_dict[str(idx)]['profile_data'],
            #              color='gray')
            # ax[-1].plot(rad_profile_dict[str(idx)]['radius_data'],
            #              rad_profile_dict[str(idx)]['profile_data'],
            #              color='gray')
            #
            # ax[idx].errorbar(rad_profile_dict[str(idx)]['radius_data'],
            #          rad_profile_dict[str(idx)]['profile_data'],
            #              yerr=rad_profile_dict[str(idx)]['profile_err'],
            #          fmt='.')
            # plt.plot(rad_profile_dict[str(idx)]['radius_data'],
            #              rad_profile_dict[str(idx)]['profile_data'],
            #              color='gray')
            # plt.show()

            # select the lower amplitude. There is a chance that the values are negative

            # plt.plot(rad_profile_dict[str(idx)]['radius_data'],
            #              rad_profile_dict[str(idx)]['profile_data'],
            #              color='gray')

            # fit
            gaussian_fit_dict = helper_func.FitTools.fit_gauss(
                x_data=rad_profile_dict[str(idx)]['radius_data'][mask_pixels_to_fit],
                y_data=rad_profile_dict[str(idx)]['profile_data'][mask_pixels_to_fit],
                y_data_err=rad_profile_dict[str(idx)]['profile_err'][mask_pixels_to_fit],
                amp_guess=max_value_in_center, mu_guess=0, sig_guess=std_pix,
                lower_amp=lower_amp, upper_amp=upper_amp,
                lower_mu=std_pix * -5, upper_mu=std_pix * 5,
                lower_sigma=std_pix, upper_sigma=std_pix * upper_sig_fact)

            # dummy_rad = np.linspace(np.min(rad_profile_dict[str(idx)]['radius_data']),
            #                         np.max(rad_profile_dict[str(idx)]['radius_data']), 500)
            # gauss = helper_func.FitTools.gaussian_func(
            #     amp=gaussian_fit_dict['amp'], mu=gaussian_fit_dict['mu'], sig=gaussian_fit_dict['sig'], x_data=dummy_rad)
            #
            # ax[idx].plot(dummy_rad, gauss)
            # plt.plot(dummy_rad, gauss)
            # plt.show()

            # get the fit results
            amp_list.append(gaussian_fit_dict['amp'])
            mu_list.append(gaussian_fit_dict['mu'])
            sig_list.append(gaussian_fit_dict['sig'])

            amp_err_list.append(gaussian_fit_dict['amp_err'])
            mu_err_list.append(gaussian_fit_dict['mu_err'])
            sig_err_list.append(gaussian_fit_dict['sig_err'])

        # plt.show()

        # get the best matching gauss
        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        mu_err_list = np.array(mu_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that make sense
        # then need to be central
        mask_mu = np.abs(mu_list) < std_pix * model_pos_rad_accept_fact
        # they must have a positive amplitude
        mask_amp = (amp_list > 0)

        # if no function was detected in the center
        if sum(mask_mu * mask_amp) == 0:
            # non detection
            mean_amp = central_apert_stats_source.max
            mean_mu = 0
            mean_sig = std_pix
            # get flux inside the 3 sigma aperture
            flux = central_apert_stats_source.sum
            # flux_err = np.sqrt(central_apert_stats_source.sum_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            flux_err = np.sqrt(central_apert_stats_source.sum_err ** 2)
            detect_flag = False
        else:
            # print(sum(mask_mu * mask_amp))
            # print(amp_list[mask_mu * mask_amp])
            # print(mu_list[mask_mu * mask_amp])
            # print(sig_list[mask_mu * mask_amp])

            mean_amp = np.mean(amp_list[mask_mu * mask_amp])
            mean_mu = np.mean(mu_list[mask_mu * mask_amp])
            mean_sig = np.mean(sig_list[mask_mu * mask_amp])

            mean_amp_err = np.mean(amp_err_list[mask_mu * mask_amp])
            mean_sig_err = np.mean(sig_err_list[mask_mu * mask_amp])

            # get gaussian integaral as flux
            flux = mean_amp * 2 * np.pi * (mean_sig ** 2)
            flux_err = np.sqrt(mean_amp_err ** 2 * (2 * mean_sig_err) ** 2)
            # flux_err = np.sqrt(flux_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = True

        dummy_rad = np.linspace(np.min(rad_profile_dict[str(idx)]['radius_data']),
                                np.max(rad_profile_dict[str(idx)]['radius_data']), 500)
        gauss = helper_func.FitTools.gaussian_func(
            amp=mean_amp, mu=mean_mu, sig=mean_sig, x_data=dummy_rad)

        photometry_dict = {
            'dummy_rad': dummy_rad,
            'gauss': gauss,
            'mean_amp': mean_amp,
            'mean_mu': mean_mu,
            'mean_sig': mean_sig,
            'flux': flux,
            'flux_err': flux_err,
            'detect_flag': detect_flag
        }
        return photometry_dict





    @staticmethod
    def measure_morph_photometry_from_img(rad_profile_dict, gauss_std, img, bkg, img_err, wcs, ra, dec):

        # get average value in the PSF aperture:
        # print(psf_dict['gaussian_fwhm'])
        # print(psf_dict['gaussian_std'])

        radius_of_interes = gauss_std * 3

        central_apert_stats_source = ApertTools.get_sky_circ_apert_stats(data=img - bkg, data_err=img_err,
                                                                               wcs=wcs,
                                                                               ra=ra, dec=dec,
                                                                               aperture_rad=radius_of_interes)
        central_apert_stats_bkg = ApertTools.get_sky_circ_apert_stats(data=bkg, data_err=img_err, wcs=wcs,
                                                                            ra=ra, dec=dec,
                                                                            aperture_rad=radius_of_interes)

        amp_list = []
        mu_list = []
        sig_list = []
        amp_err_list = []
        mu_err_list = []
        sig_err_list = []

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=len(rad_profile_dict['slit_profile_dict']['list_angle_idx']) +1)

        for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:
            mask_center = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] > gauss_std * 3 * -1) &
                           (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] < gauss_std * 3))
            min_value_in_center = np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])
            max_value_in_center = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])

            # ax[idx].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # ax[-1].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            #
            # ax[idx].errorbar(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #          rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              yerr=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'],
            #          fmt='.')
            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # plt.show()

            mask_central_pixels = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] >
                                    gauss_std * -3) &
                                   (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] <
                                    gauss_std * 3))
            # ax[idx].scatter(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
            #              color='red')

            # select the lower amplitude. There is a chance that the values are negative
            lower_amp = min_value_in_center
            upper_amp = max_value_in_center + np.abs(max_value_in_center * 2)

            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')

            gaussian_fit_dict = helper_func.FitTools.fit_gauss(
                x_data=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
                y_data=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
                y_data_err=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'][mask_central_pixels],
                amp_guess=max_value_in_center, mu_guess=0, sig_guess=gauss_std,
                lower_amp=lower_amp, upper_amp=upper_amp,
                lower_mu=gauss_std * -5, upper_mu=gauss_std * 5,
                lower_sigma=gauss_std, upper_sigma=gauss_std * 5)

            # dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
            #                         np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
            # gauss = helper_func.FitTools.gaussian_func(
            #     amp=gaussian_fit_dict['amp'], mu=gaussian_fit_dict['mu'], sig=gaussian_fit_dict['sig'], x_data=dummy_rad)
            #
            # # ax[idx].plot(dummy_rad, gauss)
            # plt.plot(dummy_rad, gauss)
            # plt.show()

            amp_list.append(gaussian_fit_dict['amp'])
            mu_list.append(gaussian_fit_dict['mu'])
            sig_list.append(gaussian_fit_dict['sig'])

            amp_err_list.append(gaussian_fit_dict['amp_err'])
            mu_err_list.append(gaussian_fit_dict['mu_err'])
            sig_err_list.append(gaussian_fit_dict['sig_err'])

        # get the best matching gauss

        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        mu_err_list = np.array(mu_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that are central
        mask_mu = np.abs(mu_list) < gauss_std * 1
        mask_amp = (amp_list > 0)

        # if no function was detected in the center
        if sum(mask_mu * mask_amp) == 0:
            # non detection
            mean_amp = central_apert_stats_source.max
            mean_mu = 0
            mean_sig = gauss_std
            # get flux inside the 3 sigma aperture
            flux = central_apert_stats_source.sum
            flux_err = np.sqrt(central_apert_stats_source.sum_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = False
        else:
            # print(sum(mask_mu * mask_amp))
            # print(amp_list[mask_mu * mask_amp])
            # print(mu_list[mask_mu * mask_amp])
            # print(sig_list[mask_mu * mask_amp])

            mean_amp = np.mean(amp_list[mask_mu * mask_amp])
            mean_mu = np.mean(mu_list[mask_mu * mask_amp])
            mean_sig = np.mean(sig_list[mask_mu * mask_amp])

            mean_amp_err = np.mean(amp_err_list[mask_mu * mask_amp])
            mean_sig_err = np.mean(sig_err_list[mask_mu * mask_amp])

            # we need to do the sigma in pixel scale though
            mean_sig_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=mean_sig, wcs=wcs)
            # get gaussian integaral as flux
            flux = mean_amp * 2 * np.pi * (mean_sig_pix ** 2)
            flux_err = np.sqrt(mean_amp_err ** 2 * (2 * mean_sig_err) ** 2)
            flux_err = np.sqrt(flux_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = True

        dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
                                np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
        gauss = helper_func.FitTools.gaussian_func(
            amp=mean_amp, mu=mean_mu, sig=mean_sig, x_data=dummy_rad)

        photometry_dict = {
            'dummy_rad': dummy_rad,
            'gauss': gauss,
            'flux': flux,
            'flux_err': flux_err,
            'detect_flag': detect_flag
        }
        return photometry_dict




class BKGTools:
    """
    all functions for background estimation
    """
    @staticmethod
    def compute_2d_bkg(data, box_size=(5, 5), filter_size=(3,3), do_sigma_clip=True, sigma=3.0, maxiters=10,
                       bkg_method='SExtractorBackground'):
        if do_sigma_clip:
            sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        else:
            sigma_clip = None

        bkg_estimator = getattr(background, bkg_method)()
        return background.Background2D(data, box_size=box_size, filter_size=filter_size, sigma_clip=sigma_clip,
                                       bkg_estimator=bkg_estimator)

    @staticmethod
    def get_scaled_bkg(ra, dec, cutout_size, bkg_cutout, bkg_wcs, scale_size_arcsec, box_size_factor=2,
                       filter_size_factor=1, do_sigma_clip=True, sigma=3.0, maxiters=10,
                       bkg_method='SExtractorBackground'):

        # estimate the bkg_box_size
        box_size = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=scale_size_arcsec * box_size_factor, wcs=bkg_wcs,
            dim=0)
        box_size = int(np.round(box_size))
        # estimate filter sie
        filter_size = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=scale_size_arcsec * filter_size_factor, wcs=bkg_wcs,
            dim=0)
        filter_size = int(np.round(filter_size))
        # filter size musst be an odd number
        if filter_size % 2 == 0:
            filter_size += 1

        bkg = BKGTools.compute_2d_bkg(data=bkg_cutout, box_size=(box_size, box_size),
                                      filter_size=(filter_size, filter_size), do_sigma_clip=do_sigma_clip,
                                      sigma=sigma, maxiters=maxiters, bkg_method=bkg_method)

        cutout_stamp_bkg = helper_func.CoordTools.get_img_cutout(
            img=bkg.background,
            wcs=bkg_wcs,
            coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=cutout_size)

        cutout_stamp_bkg_rms = helper_func.CoordTools.get_img_cutout(
            img=bkg.background_rms,
            wcs=bkg_wcs,
            coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=cutout_size)

        return cutout_stamp_bkg, cutout_stamp_bkg_rms

    @staticmethod
    def get_bkg_from_annulus(data, data_err, wcs, ra, dec, annulus_rad_in, annulus_rad_out, do_sigma_clip=True,
                             sigma=3.0, maxiters=5):
        if do_sigma_clip:
            sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        else:
            sigma_clip = None
        mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(data_err)) | (np.isnan(data_err)))
        pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        annulus_aperture = SkyCircularAnnulus(pos, r_in=annulus_rad_in * u.arcsec, r_out=annulus_rad_out * u.arcsec)
        return ApertureStats(data, annulus_aperture, error=data_err, wcs=wcs, sigma_clip=sigma_clip, mask=mask,
                                  sum_method='exact')

    @staticmethod
    def extract_bkg_from_circ_aperture(data, data_err, wcs, ra, dec, aperture_rad):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        ra : float
        dec : float
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        aper_stat : ``photutils.aperture.ApertureStats``
        """

        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        apertures = SkyCircularAperture(pos, aperture_rad * u.arcsec)

        return ApertureStats(data, apertures, wcs=wcs, error=data_err)


class ApertTools:
    """
    Class for aperture photometry
    """
    @staticmethod
    def get_ap_corr(obs, band, target=None):
        if obs=='hst':
            return phys_params.hst_broad_band_aperture_4px_corr[target][band]
        elif obs == 'hst_ha':
            return phys_params.hst_ha_aperture_4px_corr[target][band]
        elif obs == 'nircam':
            return phys_params.nircam_aperture_corr[band]['ap_corr']
        elif obs == 'miri':
            return -2.5*np.log10(2)

    @staticmethod
    def get_ap_rad(obs, band, wcs):
        if (obs == 'hst') | ( obs == 'hst_ha'):
            return wcs.proj_plane_pixel_scales()[0].value * 3600 * 4
        if obs == 'nircam':
            return wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_aperture_corr[band]['n_pix']
        if obs == 'miri':
            return phys_params.miri_aperture_rad[band]

    @staticmethod
    def get_annulus_rad(obs, band=None, wcs=None):
        if (obs == 'hst') | ( obs == 'hst_ha'):
            return (wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.hst_bkg_annulus_radii_pix['in'],
                    wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.hst_bkg_annulus_radii_pix['out'])
        if obs == 'nircam':
            return (wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_bkg_annulus_radii_pix['in'],
                    wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_bkg_annulus_radii_pix['out'])
        if obs == 'miri':
            return (phys_params.miri_bkg_annulus_radii_arcsec[band]['in'],
                    phys_params.miri_bkg_annulus_radii_arcsec[band]['out'])

    @staticmethod
    def compute_miri_photometry_aprt_corr(band, data, data_err, wcs, ra, dec, box_size=(20, 20), filter_size=(3,3),
                                          do_bkg_sigma_clip=True, bkg_sigma=3.0, bkg_maxiters=10,
                                          bkg_method='SExtractorBackground'):

        # make sure that the data provided is large enough to compute a background
        if (data.shape[0] < 5 * box_size[0]) | (data.shape[1] < 5 * box_size[1]):
            raise KeyError(data.shape, ' is the shape of the input data and should be at least 5 times larger '
                                       'than the box size to estimate the background, which is set to: ', box_size)

        # get background
        bkg_2d = PhotTools.compute_2d_bkg(data=data, box_size=box_size, filter_size=filter_size,
                                       do_sigma_clip=do_bkg_sigma_clip, sigma=bkg_sigma, maxiters=bkg_maxiters,
                                       bkg_method=bkg_method)
        # get fwhm ee radius
        rad = phys_params.miri_empirical_ee_apertures_arcsec[band]['FWHM']/2

        flux_in_50ee_rad, flux_in_50ee_rad_err = PhotTools.extract_flux_from_circ_aperture(data=data - bkg_2d.background,
                                                                                       data_err=data_err,
                                                                                       wcs=wcs,
                                                                                       ra=ra, dec=dec,
                                                                                       aperture_rad=rad)

        # import matplotlib.pyplot as plt
        # plt.imshow(data)
        # plt.show()
        # get BKG estimation
        bkg_stats = PhotTools.extract_bkg_from_circ_aperture(data=data, data_err=data_err, wcs=wcs, ra=ra, dec=dec,
                                                            aperture_rad=rad)
        # now multiply it by the ee factor
        total_flux = flux_in_50ee_rad / phys_params.miri_empirical_ee_apertures_arcsec[band]['ee']
        total_flux_err = np.sqrt((flux_in_50ee_rad_err * 2) ** 2 + (bkg_stats.std * 2) ** 2)
        # compute also median and std background
        return total_flux, total_flux_err, bkg_stats.median

    @staticmethod
    def extract_flux_from_circ_aperture(data, data_err, wcs, ra, dec, aperture_rad):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        flux : float
        flux_err : float
        """

        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        apertures = SkyCircularAperture(pos, aperture_rad * u.arcsec)
        if data_err is None:
            mask = ((np.isinf(data)) | (np.isnan(data)))
        else:
            mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(data_err)) | (np.isnan(data_err)))

        phot = aperture_photometry(data, apertures, wcs=wcs, error=data_err, mask=mask)

        flux = phot['aperture_sum'].value[0]
        if data_err is None:
            flux_err = None
        else:
            flux_err = phot['aperture_sum_err'].value[0]

        return flux, flux_err

    @staticmethod
    def get_sky_circ_apert_stats(data, data_err, wcs, ra, dec, aperture_rad):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        flux : float
        flux_err : float
        """

        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        apertures = SkyCircularAperture(pos, aperture_rad * u.arcsec)
        if data_err is None:
            mask = ((np.isinf(data)) | (np.isnan(data)))
        else:
            mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(data_err)) | (np.isnan(data_err)))

        return ApertureStats(data, apertures, wcs=wcs, error=data_err, sigma_clip=None, mask=mask)


    @staticmethod
    def get_circ_apert_stats(data, data_err, x_pos, y_pos, aperture_rad):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        data_err : ``numpy.ndarray`` or None
        x_pos, y_pos : `float
        aperture_rad : float

        Returns
        -------
        flux : float
        flux_err : float
        """

        # pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        apertures = CircularAperture((x_pos, y_pos), aperture_rad)
        if data_err is None:
            mask = ((np.isinf(data)) | (np.isnan(data)))
        else:
            mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(data_err)) | (np.isnan(data_err)))

        return ApertureStats(data, apertures, error=data_err, sigma_clip=None, mask=mask)

    @staticmethod
    def extract_flux_from_circ_aperture_jimena(ra, dec, data, err, wcs, aperture_rad, annulus_rad_in,
                                               annulus_rad_out):
        mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(err)) | (np.isnan(err)))

        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        coords_pix = wcs.world_to_pixel(pos)

        positions_sk_xp1 = wcs.pixel_to_world(coords_pix[0] + 1, coords_pix[1])
        positions_sk_xl1 = wcs.pixel_to_world(coords_pix[0] - 1, coords_pix[1])
        positions_sk_yp1 = wcs.pixel_to_world(coords_pix[0], coords_pix[1] + 1)
        positions_sk_yl1 = wcs.pixel_to_world(coords_pix[0], coords_pix[1] - 1)

        apertures = SkyCircularAperture(pos, aperture_rad * u.arcsec)
        apertures_xp1 = SkyCircularAperture(positions_sk_xp1, aperture_rad * u.arcsec)
        apertures_xl1 = SkyCircularAperture(positions_sk_xl1, aperture_rad * u.arcsec)
        apertures_yp1 = SkyCircularAperture(positions_sk_yp1, aperture_rad * u.arcsec)
        apertures_yl1 = SkyCircularAperture(positions_sk_yl1, aperture_rad * u.arcsec)

        annulus_aperture = SkyCircularAnnulus(pos, r_in=annulus_rad_in * u.arcsec, r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_xp1 = SkyCircularAnnulus(positions_sk_xp1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_xl1 = SkyCircularAnnulus(positions_sk_xl1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_yp1 = SkyCircularAnnulus(positions_sk_yp1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_yl1 = SkyCircularAnnulus(positions_sk_yl1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)

        pixel_scale = wcs.proj_plane_pixel_scales()[0].value * 3600

        annulus_aperture_xy = CircularAnnulus(coords_pix, annulus_rad_in / pixel_scale,
                                              annulus_rad_out / pixel_scale)
        annulus_masks = annulus_aperture_xy.to_mask(method='exact')
        sigclip = SigmaClip(sigma=3.0, maxiters=5)

        phot = aperture_photometry(data, apertures, wcs=wcs, error=err, mask=mask)
        aper_stats = ApertureStats(data, apertures, wcs=wcs, error=err, sigma_clip=None, mask=mask)
        bkg_stats = ApertureStats(data, annulus_aperture, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                  sum_method='exact')
        # bkg_stats_2 = ApertureStats(data, annulus_aperture, error=err, wcs=w, sigma_clip=None, mask=mask)

        phot_xp1 = aperture_photometry(data, apertures_xp1, wcs=wcs, error=err, mask=mask)
        phot_xl1 = aperture_photometry(data, apertures_xl1, wcs=wcs, error=err, mask=mask)
        phot_yp1 = aperture_photometry(data, apertures_yp1, wcs=wcs, error=err, mask=mask)
        phot_yl1 = aperture_photometry(data, apertures_yl1, wcs=wcs, error=err, mask=mask)

        bkg_stats_xp1 = ApertureStats(data, annulus_aperture_xp1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_xl1 = ApertureStats(data, annulus_aperture_xl1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_yp1 = ApertureStats(data, annulus_aperture_yp1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_yl1 = ApertureStats(data, annulus_aperture_yl1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')

        bkg_median = bkg_stats.median
        #         area_aper = aper_stats.sum_aper_area.value
        #         # area_aper[np.isnan(area_aper)] = 0
        #         area_sky = bkg_stats.sum_aper_area.value
        # #         area_sky[np.isnan(area_sky)] = 0
        #         total_bkg = bkg_median * area_aper
        #
        #         flux_dens = (phot['aperture_sum'] - total_bkg)
        #
        #         return flux_dens

        # bkg_median[np.isnan(bkg_median)] = 0
        #
        bkg_median_xp1 = bkg_stats_xp1.median
        # bkg_median_xp1[np.isnan(bkg_median_xp1)] = 0
        bkg_median_xl1 = bkg_stats_xl1.median
        # bkg_median_xl1[np.isnan(bkg_median_xl1)] = 0
        bkg_median_yp1 = bkg_stats_yp1.median
        # bkg_median_yp1[np.isnan(bkg_median_yp1)] = 0
        bkg_median_yl1 = bkg_stats_yl1.median
        # bkg_median_yl1[np.isnan(bkg_median_yl1)] = 0

        # bkg_10 = []
        # bkg_90 = []
        # bkg_10_clip = []
        # bkg_90_clip = []
        # N_pixels_annulus = []
        # N_pixel_annulus_clipped = []

        # N_pixels_aperture=[]

        # we want the range of bg values, to estimate the range of possible background levels and the uncertainty in the background
        annulus_data = annulus_masks.multiply(data)
        # print(annulus_data)
        # print(annulus_masks.data)
        if annulus_data is not None:
            annulus_data_1d = annulus_masks.multiply(data)[
                (annulus_masks.multiply(data) != 0) & (np.isfinite(annulus_masks.multiply(data))) & (
                    ~np.isnan(annulus_masks.multiply(data)))]
            if len(annulus_data_1d) > 0:
                # annulus_data=annulus_data[~np.isnan(annulus_data) & ~np.isinf(annulus_data)]
                annulus_data_filtered = sigclip(annulus_data_1d, masked=False)
                bkg_low, bkg_hi = np.quantile(annulus_data_1d,
                                              [0.1, 0.9])  # the 10% and 90% values among the bg pixel values
                bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1, 0.9])
                bkg_10 = bkg_low
                bkg_90 = bkg_hi
                bkg_10_clip = bkg_low_clip
                bkg_90_clip = bkg_hi_clip
                # annulus_data_1d = annulus_data[mask_an.data > 0]
                N_pixels_annulus = len(annulus_data_1d)
                # N_pixel_annulus_clipped = len(annulus_data_1d)-len(annulus_data_filtered)
            else:
                bkg_low = 0.
                bkg_hi = 0.
                bkg_10 = bkg_low
                bkg_90 = bkg_hi
                bkg_10_clip = 0.
                bkg_90_clip = 0.
                # annulus_data_1d = annulus_data[mask_an.data > 0]
                N_pixels_annulus = 0
        else:
            bkg_low = 0.
            bkg_hi = 0.  # the 10% and 90% values among the bg pixel values
            # bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1,0.9])
            bkg_10 = bkg_low
            bkg_90 = bkg_hi
            bkg_10_clip = 0
            bkg_90_clip = 0
            # annulus_data_1d = annulus_data[mask_an.data > 0]
            N_pixels_annulus = 0

        # bkg_10=0.1*bkg_stats_2.sum
        # bkg_90=0.9*bkg_stats_2.sum
        area_aper = aper_stats.sum_aper_area.value
        # area_aper[np.isnan(area_aper)] = 0
        area_sky = bkg_stats.sum_aper_area.value
        # area_sky[np.isnan(area_sky)] = 0
        total_bkg = bkg_median * area_aper
        total_bkg_xp1 = bkg_median_xp1 * area_aper
        total_bkg_xl1 = bkg_median_xl1 * area_aper
        total_bkg_yp1 = bkg_median_yp1 * area_aper
        total_bkg_yl1 = bkg_median_yl1 * area_aper

        total_bkg_10 = bkg_10 * area_aper
        total_bkg_90 = bkg_90 * area_aper
        total_bkg_10_clip = bkg_10_clip * area_aper
        total_bkg_90_clip = bkg_90_clip * area_aper

        bkg_std = bkg_stats.std
        # bkg_std[np.isnan(bkg_std)] = 0

        flux_dens = (phot['aperture_sum'] - total_bkg)

        flux_dens_xp1 = (phot_xp1['aperture_sum'] - total_bkg_xp1)
        flux_dens_xl1 = (phot_xl1['aperture_sum'] - total_bkg_xl1)
        flux_dens_yp1 = (phot_yp1['aperture_sum'] - total_bkg_yp1)
        flux_dens_yl1 = (phot_yl1['aperture_sum'] - total_bkg_yl1)

        flux_err_delta_apertures = np.sqrt(((flux_dens - flux_dens_xp1) ** 2 + (flux_dens - flux_dens_xl1) ** 2 + (
                flux_dens - flux_dens_yp1) ** 2 + (flux_dens - flux_dens_yl1) ** 2) / 4.)

        flux_dens_bkg_10 = (phot['aperture_sum'] - total_bkg_10)
        flux_dens_bkg_90 = (phot['aperture_sum'] - total_bkg_90)
        flux_dens_bkg_10_clip = (phot['aperture_sum'] - total_bkg_10_clip)
        flux_dens_bkg_90_clip = (phot['aperture_sum'] - total_bkg_90_clip)

        flux_dens_err = np.sqrt(pow(phot['aperture_sum_err'], 2.) + (
                pow(bkg_std * area_aper, 2) / bkg_stats.sum_aper_area.value) * np.pi / 2)
        # flux_dens_err_ir=np.sqrt(pow(phot['aperture_sum_err'],2.)+(pow(bkg_std*area_aper,2)/bkg_stats.sum_aper_area.value)*np.pi/2)/counts
        # sigma_bg times sqrt(pi/2) times aperture_area
        # phot_ap_error=np.sqrt(pow(phot['aperture_sum_err'],2.))/counts
        # err_bkg=np.sqrt((pow(bkg_std*area_aper,2)/bkg_stats.sum_aper_area.value)*np.pi/2)/counts
        # delta_90_10=(flux_dens_bkg_10 - flux_dens_bkg_90)
        # delta_90_10_clip=(flux_dens_bkg_10_clip - flux_dens_bkg_90_clip)
        flux_dens_err_9010 = np.sqrt(flux_dens_err ** 2 + (flux_dens_bkg_10 - flux_dens_bkg_90) ** 2)
        flux_dens_err_9010_clip = np.sqrt(flux_dens_err ** 2 + (flux_dens_bkg_10_clip - flux_dens_bkg_90_clip) ** 2)

        return flux_dens, flux_dens_err, flux_dens_err_9010, flux_dens_err_9010_clip, flux_err_delta_apertures

    @staticmethod
    def compute_phot_jimena(ra, dec, data, err, wcs, obs, band, aperture_rad=None, annulus_rad_in=None,
                            annulus_rad_out=None, target=None, gal_ext_corr=False):
        if aperture_rad is None:
            aperture_rad = PhotTools.get_ap_rad(obs=obs, band=band, wcs=wcs)

        if (annulus_rad_in is None) | (annulus_rad_out is None):
            annulus_rad_in, annulus_rad_out = PhotTools.get_annulus_rad(obs=obs, wcs=wcs, band=band)

        flux, flux_err, flux_err_9010, flux_err_9010_clip, flux_err_delta_apertures = \
            PhotTools.extract_flux_from_circ_aperture_jimena(
                ra=ra, dec=dec, data=data, err=err, wcs=wcs, aperture_rad=aperture_rad,
                annulus_rad_in=annulus_rad_in,
                annulus_rad_out=annulus_rad_out)
        if gal_ext_corr:
            fore_ground_ext = DustTools.get_target_gal_ext_band(target=target, obs=obs, band=band)
            flux *= 10 ** (fore_ground_ext / -2.5)

        return {'flux': flux.value[0], 'flux_err': flux_err.value[0], 'flux_err_9010': flux_err_9010.value[0],
                'flux_err_9010_clip': flux_err_9010_clip.value[0],
                'flux_err_delta_apertures': flux_err_delta_apertures.value[0]}

    @staticmethod
    def compute_ap_corr_phot_jimena(target, ra, dec, data, err, wcs, obs, band):
        flux_dict = PhotTools.compute_phot_jimena(ra=ra, dec=dec, data=data, err=err, wcs=wcs, obs=obs, band=band,
                                                  target=target)
        aperture_corr = PhotTools.get_ap_corr(obs=obs, band=band, target=target)

        flux_dict['flux'] *= 10 ** (aperture_corr / -2.5)

        return flux_dict

    @staticmethod
    def extract_flux_from_circ_aperture_sinan(X, Y, image, annulus_r_in, annulus_r_out, aperture_radii):

        """
        This function was adapted to meet some standard ike variable naming. The functionality is untouched

        Calculate the aperture photometry of given (X,Y) coordinates

        Parameters
        ----------

        annulus_r_in:
            the inner radius of the annulus at which to calculate the background
        annulus_r_out: the outer radius of the annulus at which to calculate the background
        aperture_radii: the list of aperture radii at which the photometry will be calculated.
                        in units of pixels.
        """

        # SETTING UP FOR APERTURE PHOTOMETRY AT THE GIVEN X-Y COORDINATES
        print('Initializing entire set of photometric apertures...')
        'Initializing entire set of photometric apertures...'
        # begin aperture photometry for DAO detections
        # first set positions

        # print(X, Y)
        # positions = (X, Y)
        # positions = [(X, Y)]
        """The below line transforms the x & y coordinate list or single entries into the form photutils expects the input to be in."""
        positions = np.column_stack((X, Y))

        # then circular apertures
        apertures = [CircularAperture(positions, r=r) for r in aperture_radii]
        """Possibly no need, but may need to uncomment the below two lines in case 
        two different annuli need to be defined"""
        # then a single annulus aperture (for background) - Brad used 7-9 pixels
        # annulus_apertures_phangs = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_in)
        # another annulus aperture for the aperture correction
        annulus_apertures_ac = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)
        # need to subtract the smaller annulus_apertures_phangs from annulus_apertures_ac
        # finally make a mask for the annulus aperture
        annulus_masks = annulus_apertures_ac.to_mask(method='center')

        """To plot the mask, uncomment below"""
        # plt.imshow(annulus_masks[0])
        # plt.colorbar()
        # plt.show()

        # FOR REFERENCE DETECTION IMAGE... determine robust, sig-clipped  median in the background annulus aperture at each detection location
        bkg_median = []
        bkg_std = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image)  # 25Feb2020 --  check whether the entire image is fed here
            annulus_data_1d = annulus_data[mask.data > 0]
            mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            bkg_std.append(std_sigclip)
        bkg_median = np.array(bkg_median)
        bkg_std = np.array(bkg_std)

        # FOR REFERENCE DETECTION IMAGE... conduct the actual aperture photometry, making measurements for the entire set of aperture radii specified above
        # PRODUCING THE SECOND TABLE PRODUCT ASSOCIATED WITH DAOFIND (CALLED 'APPHOT' TABLE)
        # print('Conducting aperture photometry in progressive aperture sizes on reference image...')
        'Conducting aperture photometry in progressive aperture sizes on reference image...'
        apphot = aperture_photometry(image, apertures)
        # FOR REFERENCE DETECTION IMAGE... add in the ra, dec, n_zero and bkg_median info to the apphot result
        # apphot['ra'] = ra
        # apphot['dec'] = dec
        apphot['annulus_median'] = bkg_median
        apphot['annulus_std'] = bkg_std
        # apphot['aper_bkg'] = apphot['annulus_median'] * aperture.area

        for l in range(len(aperture_radii)):
            # FOR REFERENCE DETECTION IMAGE... background subtract the initial photometry
            apphot['aperture_sum_' + str(l) + '_bkgsub'] = apphot['aperture_sum_' + str(l)] - (
                        apphot['annulus_median'] * apertures[l].area)

        # obj_list.append(np.array(apphot))

        """convert to pandas dataframe here - note that apphot.colnames & radii are local parameters """

        structure_data = np.array(apphot)
        print('Number of structures: ', structure_data.shape[0])

        structure_data_arr = np.zeros(shape=(structure_data.shape[0], len(apphot.colnames)))

        """Note that the majority of the operations around here are to convert the mildly awful astropy
        table format to a pandas dataframe"""

        for arr_x in range(structure_data.shape[0]):
            for arr_y in range(len(apphot.colnames)):
                structure_data_arr[arr_x][arr_y] = structure_data[apphot.colnames[arr_y]][arr_x]

        structure_df = pd.DataFrame(structure_data_arr, columns=apphot.colnames, dtype=np.float32)

        return structure_df


class SrcTools:
    """
    Class to gather source detection algorithms
    """

    @staticmethod
    def detect_star_like_src(data, detection_threshold, src_fwhm_pix):

        # define DAO star finder class
        dao_find = DAOStarFinder(threshold=detection_threshold, fwhm=src_fwhm_pix)
        return dao_find(data)

    @staticmethod
    def detect_star_like_src_from_topo_dict(topo_dict, src_threshold_detect_factor=3, src_fwhm_detect_factor=1):

        # perform source detection
        dao_detection = SrcTools.detect_star_like_src(
            data=topo_dict['img'] - topo_dict['bkg'],
            detection_threshold=src_threshold_detect_factor * np.nanmedian(topo_dict['bkg_rms']),
            src_fwhm_pix=src_fwhm_detect_factor * topo_dict['psf_fwhm_pix'])
        # get detected sources in
        if dao_detection is None:
            x_src = []
            y_src = []
            ra_src = []
            dec_src = []
        else:
            x_src = list(dao_detection['xcentroid'])
            y_src = list(dao_detection['ycentroid'])
            positions_world = topo_dict['wcs'].pixel_to_world(
                dao_detection['xcentroid'], dao_detection['ycentroid'])
            ra_src = list(positions_world.ra.deg)
            dec_src = list(positions_world.dec.deg)

        src_dict = {'x_src': x_src, 'y_src': y_src, 'ra_src': ra_src, 'dec_src': dec_src}

        return src_dict

    @staticmethod
    def re_center_src(init_ra, init_dec, wcs, mask, src_dict, re_center_rad_arcsec):

        init_pos = SkyCoord(ra=init_ra*u.deg, dec=init_dec*u.deg)
        init_pos_pix = wcs.world_to_pixel(init_pos)
        # check if position is masked
        print(mask.shape)

        print(sum(mask))
        # exit()
        print(np.rint(init_pos_pix[1]), np.rint(init_pos_pix[0]))
        pos_flag = mask[int(np.rint(init_pos_pix[1])), int(np.rint(init_pos_pix[0]))]
        print(pos_flag)
        exit()
        if src_dict['ra_src']:
            pos_src = SkyCoord(ra=src_dict['ra_src']*u.deg, dec=src_dict['dec_src']*u.deg)
            separation = pos_src.separation(init_pos)
            mask_src_inside_psf_fwhm = separation < re_center_rad_arcsec * u.arcsec
            if sum(mask_src_inside_psf_fwhm) == 0:
                ra_src_recenter, dec_src_recenter = init_ra, init_dec
                x_src_recenter, y_src_recenter = init_pos_pix
            else:
                mask_closest_src = separation == np.min(separation)
                ra_src_recenter = float(np.array(src_dict['ra_src'])[mask_closest_src])
                dec_src_recenter = float(np.array(src_dict['dec_src'])[mask_closest_src])
                x_src_recenter = float(np.array(src_dict['x_src'])[mask_closest_src])
                y_src_recenter = float(np.array(src_dict['y_src'])[mask_closest_src])

        else:
            ra_src_recenter, dec_src_recenter = init_ra, init_dec
            x_src_recenter, y_src_recenter = init_pos_pix

        re_center_dict = {'ra_src_recenter': ra_src_recenter, 'dec_src_recenter': dec_src_recenter,
                          'x_src_recenter': x_src_recenter, 'y_src_recenter': y_src_recenter}

        return re_center_dict

    @staticmethod
    def recenter_src_from_topo_dict(topo_dict, src_threshold_detect_factor=3, src_fwhm_detect_factor=1,
                                    re_center_rad_fact=1):

        src_dict = SrcTools.detect_star_like_src_from_topo_dict(
            topo_dict=topo_dict, src_threshold_detect_factor=src_threshold_detect_factor,
            src_fwhm_detect_factor=src_fwhm_detect_factor)

        re_center_dict = SrcTools.re_center_src(
            init_ra=topo_dict['ra'], init_dec=topo_dict['dec'], wcs=topo_dict['wcs'], mask=topo_dict['mask_bad_pixels'], src_dict=src_dict,
            re_center_rad_arcsec=re_center_rad_fact * topo_dict['psf_fwhm'])

        return re_center_dict

class PhotToolsOld:
    """
    all functions related to photometry
    """


    @staticmethod
    def extract_flux_from_circ_aperture_save(data, wcs, pos, aperture_rad, data_err=None):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        flux : float
        flux_err : float
        """
        # estimate background
        bkg = sep.Background(np.array(data, dtype=float))
        # get radius in pixel scale
        pix_radius = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=aperture_rad, wcs=wcs, dim=1)
        # pix_radius_old = (wcs.world_to_pixel(pos)[0] -
        #               wcs.world_to_pixel(SkyCoord(ra=pos.ra + aperture_rad * u.arcsec, dec=pos.dec))[0])
        # print(pix_radius)
        # print(pix_radius_old)
        # exit()
        # get the coordinates in pixel scale
        pixel_coords = wcs.world_to_pixel(pos)

        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        if data_err is None:
            bkg_rms = bkg.rms()
            data_err = np.array(bkg_rms.byteswap().newbyteorder(), dtype=float)
        else:
            data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        flux, flux_err, flag = sep.sum_circle(data=data - bkg.globalback, x=np.array([float(pixel_coords[0])]),
                                              y=np.array([float(pixel_coords[1])]), r=np.array([float(pix_radius)]),
                                              err=data_err)

        return float(flux), float(flux_err)

    # @staticmethod
    # def compute_photo_ew(wave_min_left_band, wave_max_left_band, wave_min_right_band, wave_max_right_band,
    #                      wave_min_narrow_band, wave_max_narrow_band, flux_left_band, flux_right_band, flux_narrowband):

    @staticmethod
    def compute_hst_photo_ew(target, left_band, right_band, narrow_band, flux_left_band, flux_right_band,
                             flux_narrow_band, flux_err_left_band, flux_err_right_band, flux_err_narrow_band):
        # get the piviot wavelength of both bands
        pivot_wave_left_band = helper_func.ObsTools.get_hst_band_wave(
            band=left_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=left_band),
            wave_estimator='pivot_wave', unit='angstrom')
        pivot_wave_right_band = helper_func.ObsTools.get_hst_band_wave(
            band=right_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=right_band),
            wave_estimator='pivot_wave', unit='angstrom')
        pivot_wave_narrow_band = helper_func.ObsTools.get_hst_band_wave(
            band=narrow_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=narrow_band),
            wave_estimator='pivot_wave', unit='angstrom')
        # get the effective width of the narrowband filter
        w_eff_narrow_band = helper_func.ObsTools.get_hst_band_wave(
            band=narrow_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=narrow_band),
            wave_estimator='w_eff', unit='angstrom')

        # now change from fluxes to flux densities
        flux_dens_left_band = flux_left_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_left_band)
        flux_dens_right_band = flux_right_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_right_band)
        flux_dens_narrow_band = flux_narrow_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_narrow_band)
        # convert also uncertainties with factor
        flux_err_dens_left_band = flux_err_left_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_left_band)
        flux_err_dens_right_band = flux_err_right_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_right_band)
        flux_err_dens_narrow_band = flux_err_narrow_band * helper_func.UnitTools.get_flux_unit_conv_fact(
            old_unit='mJy', new_unit='erg A-1 cm-2 s-1', pixel_size=None, band_wave=pivot_wave_narrow_band)

        # calculate the weighted continuum flux
        weight_left_band = (pivot_wave_narrow_band - pivot_wave_left_band) / (pivot_wave_right_band - pivot_wave_left_band)
        weight_right_band = (pivot_wave_right_band - pivot_wave_narrow_band) / (pivot_wave_right_band - pivot_wave_left_band)
        weighted_continuum_flux_dens = weight_left_band * flux_dens_left_band + weight_right_band * flux_dens_right_band
        # error propagation
        weighted_continuum_flux_err_dens = np.sqrt(flux_err_dens_left_band ** 2 + flux_err_dens_right_band ** 2)

        # EW estimation taken from definition at https://en.wikipedia.org/wiki/Equivalent_width
        # be aware that the emission features have negative and absorption features have positive EW!
        ew = ((weighted_continuum_flux_dens - flux_dens_narrow_band) / weighted_continuum_flux_dens) * w_eff_narrow_band
        # uncertainty estimated via error propagation if this is not clear to you look here:
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        er_err = np.sqrt(((w_eff_narrow_band * flux_err_dens_narrow_band) / weighted_continuum_flux_dens) ** 2 +
                         ((flux_dens_narrow_band * w_eff_narrow_band * weighted_continuum_flux_err_dens) /
                          (weighted_continuum_flux_dens ** 2)) ** 2)

        return ew, er_err
























