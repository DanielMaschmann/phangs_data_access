"""
Tools for spectrsocopic analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util

from astropy import constants as const
speed_of_light_kmps = const.c.to('km/s').value
from scipy.constants import c as speed_of_light_mps

from os import path

from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
from urllib import request
from TardisPipeline.readData.MUSE_WFM import get_MUSE_polyFWHM
from phangs_data_access import phys_params
from phangs_data_access.em_line_fit import FitModels

class SpecTools:
    def __init__(self):
        pass

    @staticmethod
    def instrument2wave_ref(instrument):
        assert (instrument in ['muse', 'manga', 'sdss'])
        if instrument == 'muse':
            return 'vac_wave'
        elif (instrument == 'manga') | (instrument == 'sdss'):
            return 'vac_wave'

    @staticmethod
    def get_inst_broad_sig(line, instrument='muse', unit='kmps'):
        assert (instrument in ['muse', 'manga', 'sdss'])
        if instrument == 'muse':
            wave = phys_params.opt_line_wave[line][SpecTools.instrument2wave_ref(instrument=instrument)]
            inst_broad_sig_wave = get_MUSE_polyFWHM(x=wave) / 2*np.sqrt(2*np.log(2))
            if unit == 'angstrom':
                return inst_broad_sig_wave
            elif unit in ['kmps', 'mps']:
                return SpecTools.conv_delta_wave2vel(line=line, delta_wave=inst_broad_sig_wave, vel_unit=unit,
                                                     line_ref=SpecTools.instrument2wave_ref(instrument=instrument))
        else:
            raise KeyError(instrument, ' not understand')

    @staticmethod
    def get_target_ned_redshift(target):
        """
        Function to get redshift from NED with astroquery
        Parameters
        ----------

        Returns
        -------
        redshift : float
        """

        from astroquery.ipac.ned import Ned
        # get the center of the target
        ned_table = Ned.query_object(target)

        return ned_table['Redshift'][0]

    @staticmethod
    def get_target_sys_vel(target=None, redshift=None, vel_unit='kmps'):
        """
        Function to get systemic velocity based on redshift or NED redshift
        the conversion is based on eq.(8) of Cappellari (2017)
        Parameters
        ----------

        Returns
        -------
        sys_vel : float
        """
        assert(vel_unit in ['kmps', 'mps'])
        speed_of_light = globals()['speed_of_light_%s' % vel_unit]
        if redshift is not None:
            return np.log(redshift + 1) * speed_of_light
        elif target is not None:
            redshift = SpecTools.get_target_ned_redshift(target=target)
            return np.log(redshift + 1) * speed_of_light
        else:
            raise KeyError(' either target or redshift must be not None!')

    @staticmethod
    def vel2redshift(vel, vel_unit='kmps'):
        """
        Function to convert spectral velocity to redshift
        ----------

        Returns
        -------
        redshift : float
        """
        assert (vel_unit in ['kmps', 'mps'])
        speed_of_light = globals()['speed_of_light_%s' % vel_unit]
        return np.exp(vel / speed_of_light) - 1

    @staticmethod
    def conv_rest_wave2obs_wave(rest_wave, vel_kmps):
        return rest_wave * (1 + vel_kmps / speed_of_light_kmps)

    @staticmethod
    def get_line_pos(line, vel_kmps=None, target=None, redshift=None, instrument='muse'):
        if vel_kmps is None:
            vel_kmps = SpecTools.get_target_sys_vel(target=target, redshift=redshift, vel_unit='kmps')
        return SpecTools.conv_rest_wave2obs_wave(
            rest_wave=phys_params.opt_line_wave[line][SpecTools.instrument2wave_ref(instrument=instrument)],
            vel_kmps=vel_kmps)

    @staticmethod
    def conv_vel2delta_wave(line, vel, vel_unit='kmps', line_ref='vac_wave'):
        assert(vel_unit in ['kmps', 'mps'])
        speed_of_light = globals()['speed_of_light_%s' % vel_unit]
        return vel / speed_of_light * phys_params.opt_line_wave[line][line_ref]

    @staticmethod
    def conv_delta_wave2vel(line, delta_wave, vel_unit='kmps', line_ref='vac_wave'):
        assert (vel_unit in ['kmps', 'mps'])
        speed_of_light = globals()['speed_of_light_%s' % vel_unit]
        return delta_wave * speed_of_light / phys_params.opt_line_wave[line][line_ref]

    @staticmethod
    def conv_helio_cen_vel2obs_line_wave(line_vel, line, vel_unit='kmps', line_ref='vac_wave'):
        return phys_params.opt_line_wave[line][line_ref] + SpecTools.conv_vel2delta_wave(line=line, vel=line_vel,
                                                                                   vel_unit=vel_unit, line_ref=line_ref)
    @staticmethod
    def conv_obs_line_wave2helio_cen_vel(obs_line_wave, line, vel_unit='kmps', line_ref='vac_wave'):
        line_offset = obs_line_wave - phys_params.opt_line_wave[line][line_ref]
        return SpecTools.conv_delta_wave2vel(line=line, delta_wave=line_offset, vel_unit=vel_unit, line_ref=line_ref)

    @staticmethod
    def compute_gauss(x_data, line, amp, mu_vel, sig_vel, vel_unit='kmps', line_ref='vac_wave'):
        pos_peak = SpecTools.conv_helio_cen_vel2obs_line_wave(line_vel=mu_vel, line=line, vel_unit=vel_unit,
                                                              line_ref=line_ref)
        sig_obs_wave = SpecTools.conv_vel2delta_wave(line=line, vel=sig_vel, vel_unit=vel_unit, line_ref=line_ref)
        return FitModels.gaussian(x_values=x_data, amp=amp, mu=pos_peak, sig=sig_obs_wave)

    @staticmethod
    def get_obs_gauss_from_fit_output(x_data, em_fit_dict, line, gauss_index, line_type='nl', vel_unit='kmps',
                                      instrument='muse'):

        amp = em_fit_dict['amp_%s_%i_gauss_%i' % (line_type, line, gauss_index)]
        mu_vel = em_fit_dict['mu_%s_gauss_%i' % (line_type, gauss_index)]
        sig_int_vel = em_fit_dict['sig_%s_gauss_%i' % (line_type, gauss_index)]

        # get instrumental broadening
        sig_inst_broad_vel = SpecTools.get_inst_broad_sig(line=line, instrument=instrument, unit='kmps')
        sig_obs_vel = np.sqrt(sig_int_vel ** 2 + sig_inst_broad_vel ** 2)
        return SpecTools.compute_gauss(x_data=x_data, line=line, amp=amp, mu_vel=mu_vel, sig_vel=sig_obs_vel,
                                       vel_unit=vel_unit, line_ref=SpecTools.instrument2wave_ref(instrument=instrument))

    @staticmethod
    def fit_ppxf2spec(spec_dict, target, sps_name='fsps', age_range=None, metal_range=None, ln_list=None,
                      n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0, search_outflow=False,
                      outflow_shift='redshift', outflow_mu_offset=400,outflow_sig=1200,
                      init_mu_nl_gauss=100,  init_sig_nl_gauss=200):
        """

        Parameters
        ----------
        spec_dict : dict
        sps_name : str
            can be fsps, galaxev or emiles



        Returns
        -------
        dict
        """

        if ln_list is None:
            ln_list = [4863, 4960, 5008, 6302, 6550, 6565, 6585, 6718, 6733]
            # ln_list = [5008, 6550, 6565, 6585]



        # spec_dict['spec_flux'] *= 1e-20
        # spec_dict['spec_flux_err'] *= 1e-20
        #
        flx = spec_dict['spec_flux'] * 1e-20
        flx_err = spec_dict['spec_flux_err'] * 1e-20


        velscale = speed_of_light_kmps * np.diff(np.log(spec_dict['lam'][-2:]))[0]  # Smallest velocity step
        # print('velscale ', velscale)
        # velscale = speed_of_light_kmps*np.log(spec_dict['lam'][1]/spec_dict['lam'][0])
        # print('velscale ', velscale)
        # velscale = 10
        spectra_muse, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam'], spec=flx,
                                                            velscale=velscale)
        spectra_muse_err, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam'],
                                                                spec=flx_err, velscale=velscale)

        lsf_dict = {"lam": spec_dict['lam'], "fwhm": spec_dict['lsf']}
        # get new wavelength array
        lam_gal = np.exp(ln_lam_gal)

        # get stellar library
        ppxf_dir = path.dirname(path.realpath(lib.__file__))
        basename = f"spectra_{sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        if not path.isfile(filename):
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        sps = lib.sps_lib(filename=filename, velscale=velscale, fwhm_gal=lsf_dict, norm_range=[5070, 5950],
                          #wave_range=None,
                          age_range=age_range, metal_range=metal_range)
        reg_dim = sps.templates.shape[1:]  # shape of (n_ages, n_metal)
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

        gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp=sps.ln_lam_temp,
                                                                  lam_range_gal=spec_dict['lam_range'],
                                                                  FWHM_gal=get_MUSE_polyFWHM,
                                                                  limit_doublets=False)

        templates = np.column_stack([stars_templates, gas_templates])

        n_star_temps = stars_templates.shape[1]
        component = [0] * n_star_temps
        for line_name in gas_names:
            if '[' in line_name:
                component += [2]
            else:
                component += [1]


        gas_component = np.array(component) > 0  # gas_component=True for gas templates
        moments = [4, 4, 4]
        # vel = speed_of_light_kmps * np.log(1 + redshift)  # eq.(8) of Cappellari (2017)
        sys_vel = SpecTools.get_target_sys_vel(target=target)
        redshift = SpecTools.get_target_ned_redshift(target=target)
        start_gas = [sys_vel, 150., 0, 0]  # starting guess
        start_star = [sys_vel, 150., 0, 0]
        start = [start_star, start_gas, start_gas]

        # mask bad values
        mask = np.invert(np.isnan(spectra_muse_err))
        spectra_muse_err[np.isnan(spectra_muse_err)] = np.nanmean(spectra_muse_err)
        spectra_muse[np.isnan(spectra_muse)] = 0

        pp = ppxf(templates=templates, galaxy=spectra_muse, noise=spectra_muse_err, velscale=velscale, start=start,
                  moments=moments, degree=4, mdegree=0, lam=lam_gal, lam_temp=sps.lam_temp,
                  reg_dim=reg_dim, component=component, gas_component=gas_component,
                  reddening=0.5, gas_reddening=0.5, gas_names=gas_names, mask=mask)

        light_weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
        light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()  # Normalize to light fractions

        ages, met = sps.mean_age_metal(light_weights)
        mass2light = sps.mass_to_light(light_weights, redshift=redshift)

        wavelength = pp.lam
        total_flux = pp.galaxy
        total_flux_err = pp.noise

        best_fit = pp.bestfit
        gas_best_fit = pp.gas_bestfit
        continuum_best_fit = best_fit - gas_best_fit

        em_flux = total_flux - continuum_best_fit

        # now fit the emission lines
        em_fit_dict = SpecTools.fit_em_lines2spec(ln_list=ln_list, target=target, wave=wavelength,
                                    em_flux=em_flux, em_flux_err=total_flux_err,
                                    n_nl_gauss=n_nl_gauss, n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
                                    x_data_format='wave', instrument='muse', blue_limit=30., red_limit=30.,
                                                  search_outflow=search_outflow,
                                                  outflow_shift=outflow_shift, outflow_mu_offset=outflow_mu_offset,
                                                  outflow_sig=outflow_sig,
                                                  init_mu_nl_gauss=init_mu_nl_gauss, init_sig_nl_gauss=init_sig_nl_gauss
        )

        # get velocity of balmer component
        sol_kin_comp = pp.sol[0]
        balmer_kin_comp = pp.sol[1]
        forbidden_kin_comp = pp.sol[2]

        h_beta_rest_air = 4861.333
        h_alpha_rest_air = 6562.819
        ha_continuum_window_left_rest_air = (6475.0, 6540.0)
        ha_continuum_window_right_rest_air = (6595.0, 6625.0)

        balmer_redshift = np.exp(balmer_kin_comp[0] / speed_of_light_kmps) - 1

        observed_h_beta = h_beta_rest_air * (1 + balmer_redshift)
        observed_h_alpha = h_alpha_rest_air * (1 + balmer_redshift)

        observed_sigma_h_alpha = (balmer_kin_comp[1] / speed_of_light_kmps) * h_alpha_rest_air
        observed_sigma_h_alpha = np.sqrt(observed_sigma_h_alpha ** 2 + get_MUSE_polyFWHM(observed_h_alpha))
        observed_sigma_h_beta = (balmer_kin_comp[1] / speed_of_light_kmps) * h_beta_rest_air
        observed_sigma_h_beta = np.sqrt(observed_sigma_h_beta ** 2 + get_MUSE_polyFWHM(observed_h_beta))

        mask_ha = (wavelength > (observed_h_alpha - 3 * observed_sigma_h_alpha)) & (
                wavelength < (observed_h_alpha + 3 * observed_sigma_h_alpha))
        mask_hb = (wavelength > (observed_h_beta - 3 * observed_sigma_h_beta)) & (
                wavelength < (observed_h_beta + 3 * observed_sigma_h_beta))
        # get ha component
        ha_line_comp = (total_flux - continuum_best_fit)[mask_ha]
        ha_line_comp_err = total_flux_err[mask_ha]

        # get the continuum component as a constant
        mask_cont_region = (((wavelength > (ha_continuum_window_left_rest_air[0] * (1 + balmer_redshift))) &
                            (wavelength < (ha_continuum_window_left_rest_air[1] * (1 + balmer_redshift)))) |
                            ((wavelength > (ha_continuum_window_right_rest_air[0] * (1 + balmer_redshift))) &
                             (wavelength < (ha_continuum_window_right_rest_air[1] * (1 + balmer_redshift)))))

        ha_cont_comp = np.nanmean(continuum_best_fit[mask_cont_region])
        ha_cont_comp_std = np.nanstd(continuum_best_fit[mask_cont_region])


        # since the wavelength size does not change much, we take the mean value
        ha_wave_comp = wavelength[mask_ha]
        delta_lambda_ha = np.mean((ha_wave_comp[1:] - ha_wave_comp[:-1]) / 2)

        # calculate the EW
        ha_ew = np.sum(((ha_cont_comp - ha_line_comp) / ha_cont_comp) * delta_lambda_ha)

        # uncertainty
        sigma_ew_segment = np.sqrt(((delta_lambda_ha * ha_line_comp_err) / ha_cont_comp)**2 +
                                   ((ha_line_comp * delta_lambda_ha * ha_cont_comp_std) / (ha_cont_comp ** 2)) ** 2)
        ha_ew_err = np.sqrt(np.sum(sigma_ew_segment ** 2))


        hb_line_comp = (total_flux - continuum_best_fit)[mask_hb]
        hb_cont_comp = continuum_best_fit[mask_hb]
        hb_wave_comp = wavelength[mask_hb]
        delta_lambda_hb = np.mean((hb_wave_comp[1:] - hb_wave_comp[:-1]) / 2)
        hb_ew = np.sum(((hb_cont_comp - hb_line_comp) / hb_cont_comp) * delta_lambda_hb)

        # gas_phase_metallicity
        flux_ha = pp.gas_flux[pp.gas_names == 'Halpha']
        flux_hb = pp.gas_flux[pp.gas_names == 'Hbeta']
        flux_nii = pp.gas_flux[pp.gas_names == '[OIII]5007_d']
        flux_oiii = pp.gas_flux[pp.gas_names == '[NII]6583_d']

        # pp.plot()

        o3n2 = np.log10((flux_oiii / flux_hb) / (flux_nii / flux_ha))
        gas_phase_met = 8.73 - 0.32 * o3n2[0]
        # plt.plot(hb_wave_comp, hb_line_comp)
        # plt.plot(hb_wave_comp, hb_cont_comp)
        # plt.show()
        # exit()
        #
        # # exit()
        # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
        # plt.plot(wavelength, continuum_best_fit)
        # plt.scatter(wavelength[left_idx_ha[0][0]], continuum_best_fit[left_idx_ha[0][0]])
        # plt.scatter(wavelength[right_idx_ha[0][0]], continuum_best_fit[right_idx_ha[0][0]])
        # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
        # plt.plot(wavelength, gas_best_fit)
        # plt.plot([observed_nii_1, observed_nii_1], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_h_alpha, observed_h_alpha], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_nii_2, observed_nii_2], [np.min(total_flux), np.max(total_flux)])
        # plt.show()
        #
        # plt.figure(figsize=(17, 6))
        # plt.subplot(111)
        # pp.plot()
        # plt.show()

        ppxf_dict = {
            'wavelength': wavelength, 'total_flux': total_flux, 'total_flux_err': total_flux_err,
            'best_fit': best_fit, 'gas_best_fit': gas_best_fit, 'continuum_best_fit': continuum_best_fit,
            'ages': ages, 'met': met, 'mass2light': mass2light,
            'pp': pp,
            'star_red': pp.dust[0]['sol'][0], 'gas_red': pp.dust[1]['sol'][0],
            'sol_kin_comp': sol_kin_comp, 'balmer_kin_comp': balmer_kin_comp, 'forbidden_kin_comp': forbidden_kin_comp,
            'ha_ew': ha_ew, 'ha_ew_err': ha_ew_err, 'hb_ew': hb_ew, 'gas_phase_met': gas_phase_met,
            'sys_vel': sys_vel, 'redshift': redshift
        }
        return ppxf_dict, em_fit_dict

    @staticmethod
    def get_line_mask(wave, line, target, instrument='muse', blue_limit=30., red_limit=30.):
        if line in (6550, 6565, 6585):
            nii_6550_observed_line = SpecTools.get_line_pos(line=6550, target=target, instrument=instrument)
            nii_6585_observed_line = SpecTools.get_line_pos(line=6585, target=target, instrument=instrument)
            return (wave > (nii_6550_observed_line - blue_limit)) & \
                   (wave < nii_6585_observed_line + red_limit)
        elif line in (6718, 6733):
            sii_6718_observed_line = SpecTools.get_line_pos(line=6718, target=target, instrument=instrument)
            sii_6733_observed_line = SpecTools.get_line_pos(line=6733, target=target, instrument=instrument)
            return (wave > (sii_6718_observed_line - blue_limit)) & \
                   (wave < sii_6733_observed_line + red_limit)
        else:
            obs_line = SpecTools.get_line_pos(line=line, target=target, instrument=instrument)
            return (wave > (obs_line - blue_limit)) & \
                   (wave < obs_line + red_limit)

    @staticmethod
    def get_multiple_line_mask(wave, ln_list, target, instrument='muse', blue_limit=30., red_limit=30.):

        multi_line_mask = np.zeros(len(wave), dtype=bool)
        if ln_list is None:
            ln_list = [4863, 4960, 5008, 6302, 6550, 6565, 6585, 6718, 6733]

        for line in ln_list:
            multi_line_mask += SpecTools.get_line_mask(wave=wave, line=line, target=target, instrument=instrument,
                                                   blue_limit=blue_limit, red_limit=red_limit)

        return multi_line_mask


    @staticmethod
    def fit_em_lines2spec(ln_list, target, wave, em_flux, em_flux_err, n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0,
                          x_data_format='wave', instrument='muse', blue_limit=30., red_limit=30., search_outflow=True,
                          outflow_shift='redshift', outflow_mu_offset=400,outflow_sig=1200,
                          init_mu_nl_gauss=100, init_sig_nl_gauss=200

    ):
        # get data
        ln_mask = SpecTools.get_multiple_line_mask(wave=wave, ln_list=ln_list, target=target, instrument=instrument,
                                                 blue_limit=blue_limit, red_limit=red_limit)
        # get systematic velocity
        sys_vel = SpecTools.get_target_sys_vel(target=target)

        dict_inst_broad = {}
        for line in ln_list:
            dict_inst_broad.update(
                {line: SpecTools.get_inst_broad_sig(line=line, instrument=instrument, unit='kmps')})

        # initialize emission line fit
        fit_model = FitModels()
        fit_model.set_up_model(x_data=wave[ln_mask], flx=em_flux[ln_mask], flx_err=em_flux_err[ln_mask],
                          n_nl_gauss=n_nl_gauss, n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
                          ln_list=ln_list, dict_inst_broad=dict_inst_broad, x_data_format=x_data_format)

        fit_param_restrict_dict_nl_gauss, fit_param_restrict_dict_nl_lorentz, fit_param_restrict_dict_bl_gauss = \
            SpecTools.get_fit_param_restrict_dict_outflow_search(target=target, n_nl_gauss=n_nl_gauss,
                                                                 n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
                                                                 balmer_ln=fit_model.balmer_ln, all_ln=fit_model.all_ln,
                                                                 wave=wave, em_flux=em_flux,
                                                                 instrument=instrument,
                                                                 search_outflow=search_outflow,
                                                                 outflow_shift=outflow_shift, outflow_mu_offset=outflow_mu_offset,outflow_sig=outflow_sig,
                                                                 init_mu_nl_gauss=init_mu_nl_gauss,  init_sig_nl_gauss=init_sig_nl_gauss
                                                                 )

        fit_param_dict = fit_model.run_fit(fit_param_restrict_dict_nl_gauss=fit_param_restrict_dict_nl_gauss,
                                      fit_param_restrict_dict_nl_lorentz=fit_param_restrict_dict_nl_lorentz,
                                      fit_param_restrict_dict_bl_gauss=fit_param_restrict_dict_bl_gauss)

        fit_param_dict.update({
            'sys_vel': sys_vel,
            'ln_list': ln_list,
            'wave': wave,
            'em_flux': em_flux,
            'em_flux_err': em_flux_err,
            'ln_mask': ln_mask,
            'dict_inst_broad': dict_inst_broad,
            'n_nl_gauss': n_nl_gauss,
            'n_nl_lorentz': n_nl_lorentz,
            'n_bl_gauss': n_bl_gauss})

        return fit_param_dict

    @staticmethod
    def estimate_line_amp(line, wave, em_flux, vel=None, target=None, redshift=None, instrument='muse', bin_rad=4):
        # get line position
        line_pos = SpecTools.get_line_pos(line=line, vel_kmps=vel, target=target, redshift=redshift, instrument=instrument)
        # get wavelength steps
        # print(np.where(wave == np.wave - line_pos))
        closest_idx  = (np.abs(wave - line_pos)).argmin()
        return np.nanmax(em_flux[closest_idx - bin_rad: closest_idx + bin_rad])


    @staticmethod
    def get_fit_param_restrict_dict_outflow_search(target, n_nl_gauss, n_nl_lorentz, n_bl_gauss, balmer_ln, all_ln,
                                                   wave, em_flux, instrument='muse',
                                                   search_outflow=True,
                                                   outflow_shift='blueshift', outflow_mu_offset=0,outflow_sig=1200,
                                     init_amp_nl_gauss_frac=1, lower_rel_amp_nl_gauss=0, upper_rel_amp_nl_gauss=2,
                                     amp_nl_gauss_floating=True,
                                     init_mu_nl_gauss=200, lower_mu_nl_gauss=-1000, upper_mu_nl_gauss=1000,
                                     mu_nl_gauss_floating=True,
                                     init_sig_nl_gauss=100, lower_sig_nl_gauss=0, upper_sig_nl_gauss=700,
                                     sig_nl_gauss_floating=True,

                                     init_amp_nl_lorentz_frac=1, lower_rel_amp_nl_lorentz=0, upper_rel_amp_nl_lorentz=2,
                                     amp_nl_lorentz_floating=True,
                                     init_x0_nl_lorentz=100, lower_x0_nl_lorentz=-1000, upper_x0_nl_lorentz=1000,
                                     x0_nl_lorentz_floating=True,
                                     init_gam_nl_lorentz=100, lower_gam_nl_lorentz=0, upper_gam_nl_lorentz=700,
                                     gam_nl_lorentz_floating=True,

                                     init_amp_bl_gauss_frac=0.1, lower_rel_amp_bl_gauss=0, upper_rel_amp_bl_gauss=0.5,
                                     amp_bl_gauss_floating=True,
                                     init_mu_bl_gauss=100, lower_mu_bl_gauss=-1000, upper_mu_bl_gauss=1000,
                                     mu_bl_gauss_floating=True,
                                     init_sig_bl_gauss=1000, lower_sig_bl_gauss=500, upper_sig_bl_gauss=4000,
                                     sig_bl_gauss_floating=True,

                                          ):
        """

        Parameters
        ----------
        init_amp_nl_gauss_frac
        lower_rel_amp_nl_gauss
        upper_rel_amp_nl_gauss
        amp_nl_gauss_floating
        init_mu_nl_gauss
        lower_mu_nl_gauss
        upper_mu_nl_gauss
        mu_nl_gauss_floating
        init_sig_nl_gauss
        lower_sig_nl_gauss
        upper_sig_nl_gauss
        sig_nl_gauss_floating
        init_amp_nl_lorentz_frac
        lower_rel_amp_nl_lorentz
        upper_rel_amp_nl_lorentz
        amp_nl_lorentz_floating
        init_x0_nl_lorentz
        lower_x0_nl_lorentz
        upper_x0_nl_lorentz
        x0_nl_lorentz_floating
        init_gam_nl_lorentz
        lower_gam_nl_lorentz
        upper_gam_nl_lorentz
        gam_nl_lorentz_floating
        init_amp_bl_gauss_frac
        lower_rel_amp_bl_gauss
        upper_rel_amp_bl_gauss
        amp_bl_gauss_floating
        init_mu_bl_gauss
        lower_mu_bl_gauss
        upper_mu_bl_gauss
        mu_bl_gauss_floating
        init_sig_bl_gauss
        lower_sig_bl_gauss
        upper_sig_bl_gauss
        sig_bl_gauss_floating

        Returns
        -------

        """
        # get systematic velocity
        sys_vel = SpecTools.get_target_sys_vel(target=target)

        # create the empty parameter dict
        fit_param_restrict_dict_nl_gauss = {}
        fit_param_restrict_dict_nl_lorentz = {}
        fit_param_restrict_dict_bl_gauss = {}

        # make sure that all initial parameters are in list format
        # narrow line gauss
        # prepare parameters for mu
        if isinstance(init_mu_nl_gauss, list):
            init_mu_nl_gauss_pos = init_mu_nl_gauss
        else:
            # get equally distributed mu position inside +/- init_mu_nl_gauss
            init_mu_nl_gauss_pos = []
            for index in range(n_nl_gauss):
                if (index == 0) & search_outflow:
                    if outflow_shift == 'blueshift':
                        mu_offset = sys_vel - outflow_mu_offset
                    elif outflow_shift == 'redshift':
                        mu_offset = sys_vel + outflow_mu_offset
                    else:
                        raise KeyError('outflow_mu_offset must be redshift or bueshift')
                    init_mu_nl_gauss_pos.append(mu_offset)
                else:
                    init_mu_nl_gauss_pos.append(sys_vel - init_mu_nl_gauss + ((2 * init_mu_nl_gauss / (n_nl_gauss + 1)) * (index + 1)))
        # boundaries
        if outflow_shift == 'blueshift':
            lower_mu_limit_outflow = - outflow_mu_offset - 500
            upper_mu_limit_outflow = - outflow_mu_offset + 500
        elif outflow_shift == 'redshift':
            lower_mu_limit_outflow = + outflow_mu_offset - 500
            upper_mu_limit_outflow = + outflow_mu_offset + 500
        else:
            raise KeyError('outflow_mu_offset must be redshift or bueshift')

        if not isinstance(lower_mu_nl_gauss, list):
            lower_mu_nl_gauss = [lower_mu_nl_gauss] * n_nl_gauss
            if search_outflow:
                lower_mu_nl_gauss[0] = lower_mu_limit_outflow
        if not isinstance(upper_mu_nl_gauss, list):
            upper_mu_nl_gauss = [upper_mu_nl_gauss] * n_nl_gauss
            if search_outflow:
                upper_mu_nl_gauss[0] = upper_mu_limit_outflow
        if not isinstance(mu_nl_gauss_floating, list):
            mu_nl_gauss_floating = [mu_nl_gauss_floating] * n_nl_gauss

        init_sig_outflow = outflow_sig
        lower_sig_limit_outflow = 300
        upper_sig_limit_outflow = 2000

        # prepare parameters for sigma
        if not isinstance(init_sig_nl_gauss, list):
            init_sig_nl_gauss = [init_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                init_sig_nl_gauss[0] = init_sig_outflow
        if not isinstance(lower_sig_nl_gauss, list):
            lower_sig_nl_gauss = [lower_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                lower_sig_nl_gauss[0] = lower_sig_limit_outflow
        if not isinstance(upper_sig_nl_gauss, list):
            upper_sig_nl_gauss = [upper_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                upper_sig_nl_gauss[0] = upper_sig_limit_outflow
        if not isinstance(sig_nl_gauss_floating, list):
            sig_nl_gauss_floating = [sig_nl_gauss_floating] * n_nl_gauss
        # prepare parameters for amplitudes
        if not isinstance(init_amp_nl_gauss_frac, list):
            init_amp_nl_gauss_frac = [init_amp_nl_gauss_frac] * n_nl_gauss
        if not isinstance(lower_rel_amp_nl_gauss, list):
            lower_rel_amp_nl_gauss = [lower_rel_amp_nl_gauss] * n_nl_gauss
        if not isinstance(upper_rel_amp_nl_gauss, list):
            upper_rel_amp_nl_gauss = [upper_rel_amp_nl_gauss] * n_nl_gauss
        if not isinstance(amp_nl_gauss_floating, list):
            amp_nl_gauss_floating = [amp_nl_gauss_floating] * n_nl_gauss

        # narrow line Lorentz
        # prepare parameters for mu
        if isinstance(init_x0_nl_lorentz, list):
            init_x0_nl_lorentz_pos = init_x0_nl_lorentz
        else:
            # get equally distributed x0 position inside +/- init_x0_nl_lorentz
            init_x0_nl_lorentz_pos = []
            for index in range(n_nl_lorentz):
                init_x0_nl_lorentz_pos.append(sys_vel - init_x0_nl_lorentz + ((2 * init_x0_nl_lorentz / (n_nl_lorentz + 1)) * (index + 1)))
        if not isinstance(lower_x0_nl_lorentz, list):
            lower_x0_nl_lorentz = [lower_x0_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_x0_nl_lorentz, list):
            upper_x0_nl_lorentz = [upper_x0_nl_lorentz] * n_nl_lorentz
        if not isinstance(x0_nl_lorentz_floating, list):
            x0_nl_lorentz_floating = [x0_nl_lorentz_floating] * n_nl_lorentz
        # prepare parameters for sigma
        if not isinstance(init_gam_nl_lorentz, list):
            init_gam_nl_lorentz = [init_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(lower_gam_nl_lorentz, list):
            lower_gam_nl_lorentz = [lower_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_gam_nl_lorentz, list):
            upper_gam_nl_lorentz = [upper_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(gam_nl_lorentz_floating, list):
            gam_nl_lorentz_floating = [gam_nl_lorentz_floating] * n_nl_lorentz
        # prepare parameters for amplitudes
        if not isinstance(init_amp_nl_lorentz_frac, list):
            init_amp_nl_lorentz_frac = [init_amp_nl_lorentz_frac] * n_nl_lorentz
        if not isinstance(lower_rel_amp_nl_lorentz, list):
            lower_rel_amp_nl_lorentz = [lower_rel_amp_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_rel_amp_nl_lorentz, list):
            upper_rel_amp_nl_lorentz = [upper_rel_amp_nl_lorentz] * n_nl_lorentz
        if not isinstance(amp_nl_lorentz_floating, list):
            amp_nl_lorentz_floating = [amp_nl_lorentz_floating] * n_nl_lorentz

        # broad line gauss
        # prepare parameters for mu
        if isinstance(init_mu_bl_gauss, list):
            init_mu_bl_gauss_pos = init_mu_bl_gauss
        else:
            # get equally distributed mu position inside +/- init_mu_bl_gauss
            init_mu_bl_gauss_pos = []
            for index in range(n_bl_gauss):
                init_mu_bl_gauss_pos.append(sys_vel - init_mu_bl_gauss + ((2 * init_mu_bl_gauss / (n_bl_gauss + 1)) * (index + 1)))
        if not isinstance(lower_mu_bl_gauss, list):
            lower_mu_bl_gauss = [lower_mu_bl_gauss] * n_bl_gauss
        if not isinstance(upper_mu_bl_gauss, list):
            upper_mu_bl_gauss = [upper_mu_bl_gauss] * n_bl_gauss
        if not isinstance(mu_bl_gauss_floating, list):
            mu_bl_gauss_floating = [mu_bl_gauss_floating] * n_bl_gauss
        # prepare parameters for sigma
        if not isinstance(init_sig_bl_gauss, list):
            init_sig_bl_gauss = [init_sig_bl_gauss] * n_bl_gauss
        if not isinstance(lower_sig_bl_gauss, list):
            lower_sig_bl_gauss = [lower_sig_bl_gauss] * n_bl_gauss
        if not isinstance(upper_sig_bl_gauss, list):
            upper_sig_bl_gauss = [upper_sig_bl_gauss] * n_bl_gauss
        if not isinstance(sig_bl_gauss_floating, list):
            sig_bl_gauss_floating = [sig_bl_gauss_floating] * n_bl_gauss
        # prepare parameters for amplitudes
        if not isinstance(init_amp_bl_gauss_frac, list):
            init_amp_bl_gauss_frac = [init_amp_bl_gauss_frac] * n_bl_gauss
        if not isinstance(lower_rel_amp_bl_gauss, list):
            lower_rel_amp_bl_gauss = [lower_rel_amp_bl_gauss] * n_bl_gauss
        if not isinstance(upper_rel_amp_bl_gauss, list):
            upper_rel_amp_bl_gauss = [upper_rel_amp_bl_gauss] * n_bl_gauss
        if not isinstance(amp_bl_gauss_floating, list):
            amp_bl_gauss_floating = [amp_bl_gauss_floating] * n_bl_gauss

        # fill all emission lines
        for gauss_index in range(n_nl_gauss):
            # add mu and sigma parameters
            fit_param_restrict_dict_nl_gauss.update({'nl_gauss_%i' % gauss_index:  {'mu': init_mu_nl_gauss_pos[gauss_index],
                                                              'lower_mu': sys_vel + lower_mu_nl_gauss[gauss_index],
                                                              'upper_mu': sys_vel + upper_mu_nl_gauss[gauss_index],
                                                              'mu_floating': mu_nl_gauss_floating[gauss_index],
                                                              'sig': init_sig_nl_gauss[gauss_index],
                                                              'lower_sig': lower_sig_nl_gauss[gauss_index],
                                                              'upper_sig': upper_sig_nl_gauss[gauss_index],
                                                              'sig_floating': sig_nl_gauss_floating[gauss_index]}})
            # add amplitude paramaeters
            for line in all_ln:
                if gauss_index == 0:
                    if line == 5008:
                        init_amp = SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) * 0.4
                    else:
                        init_amp = SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) * 0.01
                else:
                    init_amp = SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) * init_amp_nl_gauss_frac[gauss_index]
                fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % gauss_index].update({
                    'amp_%i' % line: init_amp,
                    'lower_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        lower_rel_amp_nl_gauss[gauss_index],
                    'upper_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        upper_rel_amp_nl_gauss[gauss_index],
                    'amp_floating_%i' % line: amp_nl_gauss_floating[gauss_index]
                })

        for lorentz_index in range(n_nl_lorentz):
            # add mu and sigma parameters
            fit_param_restrict_dict_nl_lorentz.update({'nl_lorentz_%i' % lorentz_index:  {'x0': init_x0_nl_lorentz_pos[lorentz_index],
                                                              'lower_x0': sys_vel + lower_x0_nl_lorentz[lorentz_index],
                                                              'upper_x0': sys_vel + upper_x0_nl_lorentz[lorentz_index],
                                                              'x0_floating': x0_nl_lorentz_floating[lorentz_index],
                                                              'gam': init_gam_nl_lorentz[lorentz_index],
                                                              'lower_gam': lower_gam_nl_lorentz[lorentz_index],
                                                              'upper_gam': upper_gam_nl_lorentz[lorentz_index],
                                                              'gam_floating': gam_nl_lorentz_floating[lorentz_index]}})
            # add amplitude paramaeters
            for line in all_ln:
                fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % lorentz_index].update({
                    'amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        init_amp_nl_lorentz_frac[lorentz_index],
                    'lower_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        lower_rel_amp_nl_lorentz[lorentz_index],
                    'upper_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        upper_rel_amp_nl_lorentz[lorentz_index],
                    'amp_floating_%i' % line: amp_nl_lorentz_floating[lorentz_index]
                })

        for gauss_index in range(n_bl_gauss):
            # add mu and sigma parameters
            fit_param_restrict_dict_bl_gauss.update({'bl_gauss_%i' % gauss_index:  {'mu': init_mu_bl_gauss_pos[gauss_index],
                                                              'lower_mu': sys_vel + lower_mu_bl_gauss[gauss_index],
                                                              'upper_mu': sys_vel + upper_mu_bl_gauss[gauss_index],
                                                              'mu_floating': mu_bl_gauss_floating[gauss_index],
                                                              'sig': init_sig_bl_gauss[gauss_index],
                                                              'lower_sig': lower_sig_bl_gauss[gauss_index],
                                                              'upper_sig': upper_sig_bl_gauss[gauss_index],
                                                              'sig_floating': sig_bl_gauss_floating[gauss_index]}})
            # add amplitude paramaeters
            for line in balmer_ln:
                fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % gauss_index].update({
                    'amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        init_amp_bl_gauss_frac[gauss_index],
                    'lower_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        lower_rel_amp_bl_gauss[gauss_index],
                    'upper_amp_%i' % line:
                        SpecTools.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) *
                        upper_rel_amp_bl_gauss[gauss_index],
                    'amp_floating_%i' % line: amp_bl_gauss_floating[gauss_index]
                })

        return fit_param_restrict_dict_nl_gauss, fit_param_restrict_dict_nl_lorentz, fit_param_restrict_dict_bl_gauss






#####################
##### Code dump #####
#####################
# started to develop an alternative fit for the Tardis pipeline
#
# def fit_tardis2spec(spec_dict, velocity, hdr, sps_name='fsps', age_range=None, metal_range=None, name='explore1'):
#     """
#
#     Parameters
#     ----------
#     spec_dict : dict
#     sps_name : str
#         can be fsps, galaxev or emiles
#
#
#
#     Returns
#     -------
#     dict
#     """
#     from os import path
#     # import ppxf.sps_util as lib
#     # from urllib import request
#     # from ppxf.ppxf import ppxf
#
#     import matplotlib.pyplot as plt
#
#     from TardisPipeline.utilities import util_ppxf, util_ppxf_stellarpops, util_sfh_quantities, util_ppxf_emlines
#     import TardisPipeline as tardis_module
#     codedir = os.path.dirname(os.path.realpath(tardis_module.__file__))
#
#     import ppxf.ppxf_util as util
#     from astropy.io import fits, ascii
#     from astropy import constants as const
#     from astropy.table import Table
#     import extinction
#
#     # tardis_path = '/home/egorov/Soft/ifu-pipeline/TardisPipeline/' # change to directory where you have installed DAP
#     ncpu = 20  # how many cpu would you like to use? (20-30 is fine for our server, but use no more than 8 for laptop)
#     # print(codedir+'/Templates/spectralTemplates/eMILES-noyoung/')
#     # exit()
#     configs = {  #'SSP_LIB': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         #'SSP_LIB_SFH': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         'SSP_LIB': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         'SSP_LIB_SFH': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         # 'SSP_LIB': codedir+'/Templates/spectralTemplates/eMILES-noyoung/',  # stellar library to use
#         'NORM_TEMP': 'LIGHT', 'REDSHIFT': velocity, 'MOM': 4, 'MC_PPXF': 0, 'PARALLEL': 1,
#         'ADEG': 12,
#         'ADEG_SFH': 12,
#         'MDEG': 0,
#         'MDEG_SFH': 0,
#         'MDEG_EMS': 24,
#         'NCPU': ncpu,
#         'ROOTNAME': name,
#         'SPECTRUM_SIZE': abs(hdr['CD1_1']) * 3600.,  # spaxel size in arcsec
#         # 'EMI_FILE': os.path.join(codedir, '/Templates/configurationTemplates/emission_lines.setup'),
#         'MC_PPXF_SFH': 10,
#         'EMI_FILE': codedir + '/Templates/configurationTemplates/emission_lines.setup',  # set of emission lines to fit
#         'SKY_LINES_RANGES': codedir + '/Templates/configurationTemplates/sky_lines_ranges.setup',
#         'OUTDIR': 'data_output/',
#         'MASK_WIDTH': 150,
#         'GAS_MOMENTS': 4}
#
#     velscale = speed_of_light_kmps * np.diff(np.log(spec_dict['lam'][-2:]))[0]  # Smallest velocity step
#     log_spec, logLam, velscale = util.log_rebin(lam=spec_dict['lam_range'], spec=spec_dict['spec_flux'],
#                                                 velscale=velscale)
#     c1 = fits.Column(name='LOGLAM', array=logLam, format='D')
#     c2 = fits.Column(name='LOGSPEC', array=log_spec, format='D')
#     t = fits.BinTableHDU.from_columns([c1, c2])
#     t.writeto('{}{}-ppxf_obsspec.fits'.format(configs['OUTDIR'], name), overwrite=True)
#     log_err, _, _ = util.log_rebin(spec_dict['lam_range'], spec_dict['spec_flux_err'], velscale=velscale)
#     ww = ~np.isfinite(log_spec) | ~np.isfinite(log_err) | (log_err <= 0)
#     log_err[ww] = 9999
#     log_spec[ww] = 0.
#     # # the DAP fitting routines expect log_spec and log_err to be 2D arrays containing N spectra,
#     # # here we add a dummy dimension since we are fitting only one spectrum
#     # # to fit more than one spectrum at the same time these lines can be easily adapted
#     log_err = np.expand_dims(log_err, axis=1)
#     log_spec = np.expand_dims(log_spec, axis=1)
#
#     # define the LSF of the MUSE data
#     LSF = get_MUSE_polyFWHM(np.exp(logLam), version="udf10")
#
#     # define the velocity scale in kms
#     velscale = (logLam[1] - logLam[0]) * speed_of_light_kmps
#
#     # this is the stellar kinematics ppxf wrapper function
#     ppxf_result = util_ppxf.runModule_PPXF(configs=configs,  #tasks='',
#                                            logLam=logLam,
#                                            log_spec=log_spec, log_error=log_err,
#                                            LSF=LSF)  #, velscale=velscale)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     # exit()
#     util_ppxf_stellarpops.runModule_PPXF_stellarpops(configs, logLam, log_spec, log_err, LSF, np.arange(1), ppxf_result)
#     masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#         configs)
#     print(masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err)
#
#     # read the output file which contains the best-fit from the emission lines fitting stage
#     ppxf_bestfit_gas = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit-emlines.fits'
#     hdu3 = fits.open(ppxf_bestfit_gas)
#     bestfit_gas = hdu3['FIT'].data["BESTFIT"][0]
#     mask = (hdu3['FIT'].data['BESTFIT'][0] == 0)
#     gas_templ = hdu3['FIT'].data["GAS_BESTFIT"][0]
#
#     ppxf_bestfit = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit.fits'
#     hdu_best_fit = fits.open(ppxf_bestfit)
#     cont_fit = hdu_best_fit['FIT'].data["BESTFIT"][0]
#
#     # # reddening = ppxf_sfh_data['REDDENING']
#     # hdu_best_fit_sfh = fits.open('data_output/explore1_ppxf-bestfit.fits')
#     # print(hdu_best_fit_sfh.info())
#     # print(hdu_best_fit_sfh[1].data.names)
#     #
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'])
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'].shape)
#     # print(logLam.shape)
#     # print(spec_dict['lam'].shape)
#     # # exit()
#     # # hdu_best_fit = fits.open('data_output/explore1_templates_SFH_info.fits')
#     # # print(hdu_best_fit.info())
#     # # print(hdu_best_fit[1].data.names)
#     # # print(hdu_best_fit[1].data['Age'])
#
#     plt.plot(spec_dict['lam'], spec_dict['spec_flux'])
#     plt.plot(np.exp(logLam), cont_fit)
#     plt.plot(np.exp(logLam), gas_templ)
#     plt.plot(np.exp(logLam), cont_fit + gas_templ)
#     plt.show()
#
#     exit()
#     # this the ppxf wrapper function to simulataneously fit the continuum plus emission lines
#     # util_ppxf_emlines.runModule_PPXF_emlines(configs,# '',
#     #                                          logLam, log_spec,
#     #                                          log_err, LSF, #velscale,
#     #                                          np.arange(1), ppxf_result)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     emlines = configs['OUTDIR'] + configs['ROOTNAME'] + '_emlines.fits'
#     with fits.open(emlines) as hdu_emis:
#         ems = Table(hdu_emis['EMLDATA_DATA'].data)
#
#     # This is to include SFH results, NOT TESTED!
#     with fits.open(configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf_SFH.fits') as hdu_ppxf_sfh:
#         ppxf_sfh_data = hdu_ppxf_sfh[1].data
#         masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#             configs)
#         reddening = ppxf_sfh_data['REDDENING']
#         st_props = masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err, reddening
#
#     exit()
#
#     return ems, st_props
#
#     spectra_muse_err, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam_range'],
#                                                             spec=spec_dict['spec_flux_err'], velscale=velscale)
#
#     # print(sum(np.isnan(spec_dict['spec_flux'])))
#     # print(sum(np.isnan(spectra_muse)))
#     #
#     # plt.plot(ln_lam_gal, spectra_muse_err)
#     # plt.show()
#
#     lsf_dict = {"lam": spec_dict['lam'], "fwhm": spec_dict['lsf']}
#     # get new wavelength array
#     lam_gal = np.exp(ln_lam_gal)
#     # goodpixels = util.determine_goodpixels(ln_lam=ln_lam_gal, lam_range_temp=spec_dict['lam_range'], z=redshift)
#     goodpixels = None
#     # goodpixels = (np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))
#     # print(sum(np.invert(np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))))
#     # print(sum(((spectra_muse > 0) & (spectra_muse < 100000000000000))))
#
#     # get stellar library
#     ppxf_dir = path.dirname(path.realpath(lib.__file__))
#     basename = f"spectra_{sps_name}_9.0.npz"
#     filename = path.join(ppxf_dir, 'sps_models', basename)
#     if not path.isfile(filename):
#         url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
#         request.urlretrieve(url, filename)
#
#     sps = lib.sps_lib(filename=filename, velscale=velscale, fwhm_gal=lsf_dict, norm_range=[5070, 5950],
#                       wave_range=None,
#                       age_range=age_range, metal_range=metal_range)
#     reg_dim = sps.templates.shape[1:]  # shape of (n_ages, n_metal)
#     stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
#
#     gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp=sps.ln_lam_temp,
#                                                               lam_range_gal=spec_dict['lam_range'],
#                                                               FWHM_gal=get_MUSE_polyFWHM)
#
#     templates = np.column_stack([stars_templates, gas_templates])
#
#     n_star_temps = stars_templates.shape[1]
#     component = [0] * n_star_temps
#     for line_name in gas_names:
#         if '[' in line_name:
#             component += [2]
#         else:
#             component += [1]
#
#     gas_component = np.array(component) > 0  # gas_component=True for gas templates
#
#     moments = [4, 4, 4]
#
#     vel = speed_of_light_kmps * np.log(1 + redshift)  # eq.(8) of Cappellari (2017)
#     start_gas = [vel, 150., 0, 0]  # starting guess
#     start_star = [vel, 150., 0, 0]
#     print(start_gas)
#     start = [start_star, start_gas, start_gas]
#
#     pp = ppxf(templates=templates, galaxy=spectra_muse, noise=spectra_muse_err, velscale=velscale, start=start,
#               moments=moments, degree=-1, mdegree=4, lam=lam_gal, lam_temp=sps.lam_temp,  #regul=1/rms,
#               reg_dim=reg_dim, component=component, gas_component=gas_component,  #reddening=0,
#               gas_names=gas_names, goodpixels=goodpixels)
#
#     light_weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
#     light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
#     light_weights /= light_weights.sum()  # Normalize to light fractions
#
#     # light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
#     # light_weights = light_weights.reshape(reg_dim)
#
#     ages, met = sps.mean_age_metal(light_weights)
#     mass2light = sps.mass_to_light(light_weights, redshift=redshift)
#
#     return {'pp': pp, 'ages': ages, 'met': met, 'mass2light': mass2light}
#
#     # wavelength = pp.lam
#     # total_flux = pp.galaxy
#     # total_flux_err = pp.noise
#     #
#     # best_fit = pp.bestfit
#     # gas_best_fit = pp.gas_bestfit
#     # continuum_best_fit = best_fit - gas_best_fit
#     #
#     # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
#     # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
#     # plt.plot(wavelength, gas_best_fit)
#     # plt.show()
#     #
#     #
#     #
#     #
#     # plt.figure(figsize=(17, 6))
#     # plt.subplot(111)
#     # pp.plot()
#     # plt.show()
#     #
#     # plt.figure(figsize=(9, 3))
#     # sps.plot(light_weights)
#     # plt.title("Light Weights Fractions");
#     # plt.show()
#     #
#     # exit()