"""
This script gathers the very core function of modeling and fitting emission line profiles
"""

# from spec_fit.basic_attributes import PhysicalConstants

from scipy.constants import c as speed_of_light_mps
# from scipy.integrate import quad

import numpy as np
from astropy.modeling.models import Voigt1D
# import time
import iminuit
from phangs_data_access import phys_params


class FitModels:
    """
    Class to gather all needed fitting models

    Models:
    for emission line profiles we use Gaussian and Lorentzian functions
    as we convolve them with the current instrumental broadening we have to convolve them with a gaussian function.
    For a Gaussian function we can simply quadratically add the sigmas.
    For a Lorentzian this results in a Voigt function. For an overview see Jain et al. 2018
    https://doi.org/10.1016/j.apsusc.2018.03.190
    https://www.sciencedirect.com/science/article/pii/S0169433218308766

    """

    def __init__(self):
        super().__init__()

        """
        Gather all parameters to set up a fit model
        """

        # Parameters to set up the model
        # wavelength, flux and flux uncertainties
        self.x_data_format = 'wave'
        self.x_data = None
        self.flx = None
        self.flx_err = None
        # list of balmer lines to fit
        self.balmer_ln = []
        # list forbidden lines
        self.forbid_ln = []
        # list of independent forbidden lines
        self.forbid_indep_ln = []
        # list of all lines
        self.all_ln = []
        # list of all independent lines
        self.all_indep_ln = []
        # instrumental broadening for each line
        self.dict_inst_broad = None
        # number models
        # narrow lines with gaussian model
        self.n_nl_gauss = 0
        # narrow lines with lorentzian model
        self.n_nl_lorentz = 0
        # broad lines with gaussian model
        self.n_bl_gauss = 0
        # line ratios which are usable for example [NII] doublet or the [OIII] doublet
        self.use_line_ratio = True

        # add all parameters to used by a fit
        self.fit_param_dict = {}
        # state for each parameter which index it has
        self.fit_param_id_dict = {}

    @staticmethod
    def gaussian(x_values, amp, mu, sig):
        """
        Simple gaussian function
        Parameters
        ----------
        x_values : array-like wavelength or velocity
        amp : float amplitude
        mu : float position
        sig : float sigma/width

        Returns
        -------
        flux : array-like the modelled flux
        """
        return amp * np.exp(-(x_values - mu) ** 2 / (2 * sig ** 2))

    @staticmethod
    def lorentzian(x_values, amp, x0, gam):
        """
        Simple Lorentzian function
        Parameters
        ----------
        x_values : array-like wavelength or velocity
        amp : float amplitude
        x0 : float position
        gam : float gamma/width

        Returns
        -------
        flux : array-like the modelled flux
        """
        return amp * gam / ((x_values - x0) ** 2 + gam ** 2)

    @staticmethod
    def voigt_tf(x_values, amp, x0, gam, sig):
        """
        Voigt function which is the result of a convolution between a gaussian and a Lorentzian.
        This function if encoded in a way that the fit parameters are tensorflow tensors
        Parameters
        ----------
        x_values : array-like wavelength or velocity
        amp : float amplitude
        x0 : float position
        gam : float gamma/width of the Lorentzian function
        sig : float sigma/width of the Gaussian function

        Returns
        -------
        flux : array-like the modelled flux
        """

        # amp.numpy() * real(wofz((x_values - x0.numpy() + 1j * gam.numpy()) / sig / sqrt(2))) / sig / sqrt(2 * pi)
        return Voigt1D(x_0=x0, amplitude_L=amp, fwhm_L=gam, fwhm_G=sig)(x_values)

    @staticmethod
    def voigt(x_values, amp, x0, gam, sig):
        """
        Voigt function which is the result of a convolution between a gaussian and a Lorentzian.
        This function if encoded in a way that the fit parameters are of type float and not tensorflow objects
        Parameters
        ----------
        x_values : array-like wavelength or velocity
        amp : float amplitude
        x0 : float position
        gam : float gamma/width of the Lorentzian function
        sig : float sigma/width of the Gaussian function

        Returns
        -------
        flux : array-like the modelled flux
        """

        # amp.numpy() * real(wofz((x_values - x0.numpy() + 1j * gam.numpy()) / sig / sqrt(2))) / sig / sqrt(2 * pi)
        return Voigt1D(x_0=x0, amplitude_L=amp, fwhm_L=gam, fwhm_G=sig)(x_values)

    def add_gauss_conv_line_model(self, line):
        """
        function to set up a gaussian function convolved with the instrumental broadening
        Parameters
        ----------
        line : int identifier of the emission line at which point the instrumental broadening was estimated
        """
        if self.x_data_format == 'wave':
            inst_broad = self.dict_inst_broad[line] / (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[line]['vac_wave']
        elif self.x_data_format == 'vel':
            inst_broad = self.dict_inst_broad[line]
        else:
            raise KeyError('self.x_data_format must be wave or vel')

        setattr(self, 'gauss_conv_line_%i' % line, lambda x_values, amp, mu, sig:
                self.gaussian(x_values=x_values, amp=amp, mu=mu,
                              sig=np.sqrt(sig ** 2 + inst_broad ** 2)))

    def add_lorentz_conv_line_model(self, line):
        """
        function to set up a lorentzian function convolved with the instrumental broadening
        Parameters
        ----------
        line : int identifier of the emission line at which point the instrumental broadening was estimated
        """
        if self.x_data_format == 'wave':
            inst_broad = self.dict_inst_broad[line] / (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[4863]['vac_wave']
        elif self.x_data_format == 'vel':
            inst_broad = self.dict_inst_broad[line]
        else:
            raise KeyError('self.x_data_format must be wave or vel')

        setattr(self, 'lorentz_conv_line_%i' % line, lambda x_values, amp, x0, gam:
                self.voigt_tf(x_values=x_values, amp=amp, x0=x0, gam=gam, sig=inst_broad))

    def set_up_model(self, x_data, flx, flx_err, n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0,
                     ln_list=None, dict_inst_broad=None, x_data_format='wave', use_line_ratio=True):

        self.x_data = x_data
        self.flx = flx
        self.flx_err = flx_err

        self.n_nl_gauss = n_nl_gauss
        self.n_nl_lorentz = n_nl_lorentz
        self.n_bl_gauss = n_bl_gauss

        self.use_line_ratio = use_line_ratio

        if ln_list is None:
            ln_list = [4863, 5008, 6550, 6565, 6585]


        for line in ln_list:
            if phys_params.opt_line_wave[line]['transition'] == 'balmer':
                self.balmer_ln.append(line)
            elif phys_params.opt_line_wave[line]['transition'] == 'forbidden':
                self.forbid_ln.append(line)

        self.all_ln = self.balmer_ln + self.forbid_ln

        # add all independent lines
        self.all_indep_ln = self.all_ln.copy()
        self.forbid_indep_ln = self.forbid_ln.copy()
        if self.use_line_ratio:
            for line in [6550, 4960, 6366]:
                if line in self.all_indep_ln:
                    self.all_indep_ln.remove(line)
                    self.forbid_indep_ln.remove(line)

        # if the instrumental broadening is not given it will be handled as 0
        if dict_inst_broad is None:
            dict_inst_broad = {}
            for line in self.all_ln:
                dict_inst_broad.update({'%i' % line: 0})

        self.dict_inst_broad = dict_inst_broad
        # set up convolved function as available models
        for line in self.balmer_ln:
            if (n_nl_gauss > 0) or (n_bl_gauss > 0):
                self.add_gauss_conv_line_model(line=line)
            if n_nl_lorentz > 0:
                self.add_lorentz_conv_line_model(line=line)

        for line in self.forbid_ln:
            if n_nl_gauss > 0:
                self.add_gauss_conv_line_model(line=line)
            if n_nl_lorentz > 0:
                self.add_lorentz_conv_line_model(line=line)

        self.x_data_format = x_data_format

    def chi2(self, model_predict):
        """

        Parameters
        ----------
        model_predict : array_like
             must be of same shape as self.flx and self.flx_err

        Returns
        -------
            returns the mean weighted chi^2 value
        """
        squared = ((model_predict - self.flx) / self.flx_err) ** 2
        return np.sum(squared)

    def calc_model_predict(self, *params):

        prediction = np.zeros(len(self.flx), dtype=float)

        for nl_gauss_index in range(self.n_nl_gauss):

            for line in self.all_ln:
                # check if line ratio is applicable
                if (line == 6550) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_gauss_%i' % (6585, nl_gauss_index)]] / phys_params.em_ratios['6585/6550']
                elif (line == 4960) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_gauss_%i' % (5008, nl_gauss_index)]] / phys_params.em_ratios['5008/4960']
                elif (line == 6366) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_gauss_%i' % (6302, nl_gauss_index)]] / phys_params.em_ratios['6302/6366']
                else:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)]]
                prediction += getattr(self, 'gauss_conv_line_%i' % line)\
                    (x_values=self.x_data,
                     amp=amp,
                     mu=(params[self.fit_param_id_dict['mu_nl_gauss_%i' % nl_gauss_index]] / (speed_of_light_mps * 1e-3) *
                         phys_params.opt_line_wave[line]['vac_wave'] + phys_params.opt_line_wave[line]['vac_wave']),
                     sig=(params[self.fit_param_id_dict['sig_nl_gauss_%i' % nl_gauss_index]] / (speed_of_light_mps * 1e-3) *
                          phys_params.opt_line_wave[line]['vac_wave']))

        for nl_lorentz_index in range(self.n_nl_lorentz):
            for line in self.all_ln:
                # check if line ratio is applicable
                if (line == 6550) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_lorentz_%i' % (6585, nl_lorentz_index)]] / phys_params.em_ratios['6585/6550']
                elif (line == 4960) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_lorentz_%i' % (5008, nl_lorentz_index)]] / phys_params.em_ratios['5008/4960']
                elif (line == 6366) & self.use_line_ratio:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_lorentz_%i' % (6302, nl_lorentz_index)]] / phys_params.em_ratios['6302/6366']
                else:
                    amp = params[self.fit_param_id_dict['amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index)]]
                prediction += getattr(self, 'lorentz_conv_line_%i' % line)\
                    (x_values=self.x_data,
                     amp=amp,
                     x0=(params[self.fit_param_id_dict['x0_nl_lorentz_%i' % nl_lorentz_index]] /
                         (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[line]['vac_wave'] +
                         phys_params.opt_line_wave[line]['vac_wave']),
                     gam=(params[self.fit_param_id_dict['gam_nl_lorentz_%i' % nl_lorentz_index]]
                          / (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[line]['vac_wave']))

        for bl_gauss_index in range(self.n_bl_gauss):
            for line in self.balmer_ln:
                prediction += getattr(self, 'gauss_conv_line_%i' % line)\
                    (x_values=self.x_data,
                     amp=params[self.fit_param_id_dict['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)]],
                     mu=(params[self.fit_param_id_dict['mu_bl_gauss_%i' % bl_gauss_index]] / (speed_of_light_mps * 1e-3) *
                         phys_params.opt_line_wave[line]['vac_wave'] + phys_params.opt_line_wave[line]['vac_wave']),
                     sig=(params[self.fit_param_id_dict['sig_bl_gauss_%i' % bl_gauss_index]] / (speed_of_light_mps * 1e-3) *
                          phys_params.opt_line_wave[line]['vac_wave']))

        return prediction

    def calc_model_predict_chi2_val(self, *params):
        return self.chi2(self.calc_model_predict(*params))

    def get_init_guess_list(self, fit_param_restrict_dict_nl_gauss,
                            fit_param_restrict_dict_nl_lorentz,
                            fit_param_restrict_dict_bl_gauss):
        # list with initial guesses
        init_guess_list = []
        # list with variable names
        param_name_list = []
        # reset param id list
        self.fit_param_id_dict = {}
        param_id = 0

        for nl_gauss_index in range(self.n_nl_gauss):
            init_guess_list.append(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['mu'])
            param_name_list.append('mu_nl_gauss_%i' % nl_gauss_index)
            self.fit_param_id_dict.update({'mu_nl_gauss_%i' % nl_gauss_index: param_id})
            param_id += 1
            init_guess_list.append(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['sig'])
            param_name_list.append('sig_nl_gauss_%i' % nl_gauss_index)
            self.fit_param_id_dict.update({'sig_nl_gauss_%i' % nl_gauss_index: param_id})
            param_id += 1
            for line in self.all_indep_ln:
                init_guess_list.append(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['amp_%i' % line])
                param_name_list.append('amp_nl_%i_gauss_%i' % (line, nl_gauss_index))
                self.fit_param_id_dict.update({'amp_nl_%i_gauss_%i' % (line, nl_gauss_index): param_id})
                param_id += 1

        for nl_lorentz_index in range(self.n_nl_lorentz):
            init_guess_list.append(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['x0'])
            param_name_list.append('x0_nl_lorentz_%i' % nl_lorentz_index)
            self.fit_param_id_dict.update({'x0_nl_lorentz_%i' % nl_lorentz_index: param_id})
            param_id += 1
            init_guess_list.append(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['gam'])
            param_name_list.append('gam_nl_lorentz_%i' % nl_lorentz_index)
            self.fit_param_id_dict.update({'gam_nl_lorentz_%i' % nl_lorentz_index: param_id})
            param_id += 1
            for line in self.all_indep_ln:
                init_guess_list.append(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['amp_%i' % line])
                param_name_list.append('amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index))
                self.fit_param_id_dict.update({'amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index): param_id})
                param_id += 1

        for bl_gauss_index in range(self.n_bl_gauss):
            init_guess_list.append(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['mu'])
            param_name_list.append('mu_bl_gauss_%i' % bl_gauss_index)
            self.fit_param_id_dict.update({'mu_bl_gauss_%i' % bl_gauss_index: param_id})
            param_id += 1
            init_guess_list.append(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['sig'])
            param_name_list.append('sig_bl_gauss_%i' % bl_gauss_index)
            self.fit_param_id_dict.update({'sig_bl_gauss_%i' % bl_gauss_index: param_id})
            param_id += 1
            for line in self.balmer_ln:
                init_guess_list.append(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['amp_%i' % line])
                param_name_list.append('amp_bl_%i_gauss_%i' % (line, bl_gauss_index))
                self.fit_param_id_dict.update({'amp_bl_%i_gauss_%i' % (line, bl_gauss_index): param_id})
                param_id += 1

        return init_guess_list, param_name_list

    def set_param_restrict(self, minimizer,
                           fit_param_restrict_dict_nl_gauss,
                           fit_param_restrict_dict_nl_lorentz,
                           fit_param_restrict_dict_bl_gauss):

        for nl_gauss_index in range(self.n_nl_gauss):
            minimizer.limits['mu_nl_gauss_%i' % nl_gauss_index] = \
                (fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['lower_mu'],
                 fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['upper_mu'])
            minimizer.fixed['mu_nl_gauss_%i' % nl_gauss_index] = np.invert(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['mu_floating'])
            minimizer.limits['sig_nl_gauss_%i' % nl_gauss_index] = \
                (fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['lower_sig'],
                 fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['upper_sig'])
            minimizer.fixed['sig_nl_gauss_%i' % nl_gauss_index] = np.invert(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['sig_floating'])
            for line in self.all_indep_ln:
                minimizer.limits['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)] = \
                    (fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['lower_amp_%i' % line],
                     fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['upper_amp_%i' % line])
                minimizer.fixed['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)] = \
                    np.invert(fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % nl_gauss_index]['amp_floating_%i' % line])

        for nl_lorentz_index in range(self.n_nl_lorentz):
            minimizer.limits['x0_nl_lorentz_%i' % nl_lorentz_index] = \
                (fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['lower_x0'],
                 fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['upper_x0'])
            minimizer.fixed['x0_nl_lorentz_%i' % nl_lorentz_index] = np.invert(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['x0_floating'])
            minimizer.limits['gam_nl_lorentz_%i' % nl_lorentz_index] = \
                (fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['lower_gam'],
                 fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['upper_gam'])
            minimizer.fixed['gam_nl_lorentz_%i' % nl_lorentz_index] = np.invert(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['gam_floating'])
            for line in self.all_indep_ln:
                minimizer.limits['amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index)] = \
                    (fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['lower_amp_%i' % line],
                     fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['upper_amp_%i' % line])
                minimizer.fixed['amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index)] = \
                    np.invert(fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % nl_lorentz_index]['amp_floating_%i' % line])

        for bl_gauss_index in range(self.n_bl_gauss):
            minimizer.limits['mu_bl_gauss_%i' % bl_gauss_index] = \
                (fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['lower_mu'],
                 fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['upper_mu'])
            minimizer.fixed['mu_bl_gauss_%i' % bl_gauss_index] = np.invert(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['mu_floating'])
            minimizer.limits['sig_bl_gauss_%i' % bl_gauss_index] = \
                (fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['lower_sig'],
                 fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['upper_sig'])
            minimizer.fixed['sig_bl_gauss_%i' % bl_gauss_index] = np.invert(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['sig_floating'])
            for line in self.balmer_ln:
                minimizer.limits['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)] = \
                    (fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['lower_amp_%i' % line],
                     fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['upper_amp_%i' % line])
                minimizer.fixed['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)] = \
                    np.invert(fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % bl_gauss_index]['amp_floating_%i' % line])

    def get_fit_result_dict(self, minimizer, fit_result_dict=None):

        if fit_result_dict is None:
            fit_result_dict = {}

        for nl_gauss_index in range(self.n_nl_gauss):
            fit_result_dict.update({'mu_nl_gauss_%i' % nl_gauss_index:
                                       minimizer.values['mu_nl_gauss_%i' % nl_gauss_index]})
            fit_result_dict.update({'mu_nl_gauss_%i_err' % nl_gauss_index:
                                       minimizer.errors['mu_nl_gauss_%i' % nl_gauss_index]})
            fit_result_dict.update({'sig_nl_gauss_%i' % nl_gauss_index:
                                       minimizer.values['sig_nl_gauss_%i' % nl_gauss_index]})
            fit_result_dict.update({'sig_nl_gauss_%i_err' % nl_gauss_index:
                                       minimizer.errors['sig_nl_gauss_%i' % nl_gauss_index]})
            for line in self.all_ln:
                if (line == 6550) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_gauss_%i' % (line, nl_gauss_index):
                                               minimizer.values['amp_nl_%i_gauss_%i' % (6585, nl_gauss_index)] /
                                               phys_params.em_ratios['6585/6550']})
                    fit_result_dict.update({'amp_nl_%i_gauss_%i_err' % (line, nl_gauss_index):
                                               minimizer.errors['amp_nl_%i_gauss_%i' % (6585, nl_gauss_index)] /
                                               phys_params.em_ratios['6585/6550']})
                elif (line == 4960) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_gauss_%i' % (line, nl_gauss_index):
                                               minimizer.values['amp_nl_%i_gauss_%i' % (5008, nl_gauss_index)] /
                                               phys_params.em_ratios['5008/4960']})
                    fit_result_dict.update({'amp_nl_%i_gauss_%i_err' % (line, nl_gauss_index):
                                               minimizer.errors['amp_nl_%i_gauss_%i' % (5008, nl_gauss_index)] /
                                               phys_params.em_ratios['5008/4960']})
                elif (line == 6366) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_gauss_%i' % (line, nl_gauss_index):
                                               minimizer.values['amp_nl_%i_gauss_%i' % (6302, nl_gauss_index)] /
                                               phys_params.em_ratios['6302/6366']})
                    fit_result_dict.update({'amp_nl_%i_gauss_%i_err' % (line, nl_gauss_index):
                                               minimizer.errors['amp_nl_%i_gauss_%i' % (6302, nl_gauss_index)] /
                                               phys_params.em_ratios['6302/6366']})
                else:
                    fit_result_dict.update({'amp_nl_%i_gauss_%i' % (line, nl_gauss_index):
                                           minimizer.values['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)]})
                    fit_result_dict.update({'amp_nl_%i_gauss_%i_err' % (line, nl_gauss_index):
                                           minimizer.errors['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)]})
                # get fluxes
                inst_broad = self.dict_inst_broad[line] / (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[line]['vac_wave']
                sig = (fit_result_dict['sig_nl_gauss_%i' % nl_gauss_index] / (speed_of_light_mps * 1e-3) *
                       phys_params.opt_line_wave[line]['vac_wave'])
                sig_conv = np.sqrt(sig ** 2 + inst_broad ** 2)
                sig_err = (fit_result_dict['sig_nl_gauss_%i_err' % nl_gauss_index] / (speed_of_light_mps * 1e-3) *
                           phys_params.opt_line_wave[line]['vac_wave'])
                flux = fit_result_dict['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)] * sig_conv * np.sqrt(2 * np.pi)
                flux_err = (np.sqrt(2 * np.pi) *
                            np.sqrt((sig_err * fit_result_dict['amp_nl_%i_gauss_%i' % (line, nl_gauss_index)]) ** 2 +
                                    (fit_result_dict['amp_nl_%i_gauss_%i_err' % (line, nl_gauss_index)] * sig_conv) ** 2))
                fit_result_dict.update({'flux_nl_%i_gauss_%i' % (line, nl_gauss_index): flux})
                fit_result_dict.update({'flux_nl_%i_gauss_%i_err' % (line, nl_gauss_index): flux_err})

        for nl_lorentz_index in range(self.n_nl_lorentz):
            fit_result_dict.update({'mu_nl_lorentz_%i' % nl_lorentz_index:
                                       minimizer.values['mu_nl_lorentz_%i' % nl_lorentz_index]})
            fit_result_dict.update({'mu_nl_lorentz_%i_err' % nl_lorentz_index:
                                       minimizer.errors['mu_nl_lorentz_%i' % nl_lorentz_index]})
            fit_result_dict.update({'sig_nl_lorentz_%i' % nl_lorentz_index:
                                       minimizer.values['sig_nl_lorentz_%i' % nl_lorentz_index]})
            fit_result_dict.update({'sig_nl_lorentz_%i_err' % nl_lorentz_index:
                                       minimizer.errors['sig_nl_lorentz_%i' % nl_lorentz_index]})
            for line in self.all_ln:
                if (line == 6550) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index):
                                               minimizer.values['amp_nl_%i_lorentz_%i' % (6585, nl_lorentz_index)] /
                                               phys_params.em_ratios['6585/6550']})
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i_err' % (line, nl_lorentz_index):
                                               minimizer.errors['amp_nl_%i_lorentz_%i' % (6585, nl_lorentz_index)] /
                                               phys_params.em_ratios['6585/6550']})
                elif (line == 4960) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index):
                                               minimizer.values['amp_nl_%i_lorentz_%i' % (5008, nl_lorentz_index)] /
                                               phys_params.em_ratios['5008/4960']})
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i_err' % (line, nl_lorentz_index):
                                               minimizer.errors['amp_nl_%i_lorentz_%i' % (5008, nl_lorentz_index)] /
                                               phys_params.em_ratios['5008/4960']})
                elif (line == 6366) & self.use_line_ratio:
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index):
                                               minimizer.values['amp_nl_%i_lorentz_%i' % (6302, nl_lorentz_index)] /
                                               phys_params.em_ratios['6302/6366']})
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i_err' % (line, nl_lorentz_index):
                                               minimizer.errors['amp_nl_%i_lorentz_%i' % (6302, nl_lorentz_index)] /
                                               phys_params.em_ratios['6302/6366']})
                else:
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index):
                                           minimizer.values['amp_nl_%i_lorentz_%i' % (line, nl_lorentz_index)]})
                    fit_result_dict.update({'amp_nl_%i_lorentz_%i_err' % (line, nl_lorentz_index):
                                           minimizer.values['amp_nl_%i_lorentz_%i_err' % (line, nl_lorentz_index)]})

        for bl_gauss_index in range(self.n_bl_gauss):
            fit_result_dict.update({'mu_bl_gauss_%i' % bl_gauss_index:
                                       minimizer.values['mu_bl_gauss_%i' % bl_gauss_index]})
            fit_result_dict.update({'mu_bl_gauss_%i_err' % bl_gauss_index:
                                       minimizer.errors['mu_bl_gauss_%i' % bl_gauss_index]})
            fit_result_dict.update({'sig_bl_gauss_%i' % bl_gauss_index:
                                       minimizer.values['sig_bl_gauss_%i' % bl_gauss_index]})
            fit_result_dict.update({'sig_bl_gauss_%i_err' % bl_gauss_index:
                                       minimizer.errors['sig_bl_gauss_%i' % bl_gauss_index]})
            for line in self.balmer_ln:
                fit_result_dict.update({'amp_bl_%i_gauss_%i' % (line, bl_gauss_index):
                                           minimizer.values['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)]})
                fit_result_dict.update({'amp_bl_%i_gauss_%i_err' % (line, bl_gauss_index):
                                           minimizer.errors['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)]})

                # get fluxes
                inst_broad = self.dict_inst_broad[line] / (speed_of_light_mps * 1e-3) * phys_params.opt_line_wave[line][
                    'vac_wave']
                sig = (fit_result_dict['sig_bl_gauss_%i' % bl_gauss_index] / (speed_of_light_mps * 1e-3) *
                       phys_params.opt_line_wave[line]['vac_wave'])
                sig_conv = np.sqrt(sig ** 2 + inst_broad ** 2)
                sig_err = (fit_result_dict['sig_bl_gauss_%i_err' % bl_gauss_index] / (speed_of_light_mps * 1e-3) *
                           phys_params.opt_line_wave[line]['vac_wave'])
                flux = fit_result_dict['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)] * sig_conv * np.sqrt(2 * np.pi)
                flux_err = (np.sqrt(2 * np.pi) *
                            np.sqrt((sig_err * fit_result_dict['amp_bl_%i_gauss_%i' % (line, bl_gauss_index)]) ** 2 +
                                    (fit_result_dict['amp_bl_%i_gauss_%i_err' % (line, bl_gauss_index)] * sig_conv) ** 2))
                fit_result_dict.update({'flux_bl_%i_gauss_%i' % (line, bl_gauss_index): flux})
                fit_result_dict.update({'flux_bl_%i_gauss_%i_err' % (line, bl_gauss_index): flux_err})



        return fit_result_dict

    def run_fit(self, fit_param_restrict_dict_nl_gauss,
                fit_param_restrict_dict_nl_lorentz,
                fit_param_restrict_dict_bl_gauss):

        init_guess_list, param_name_list = self.get_init_guess_list(
            fit_param_restrict_dict_nl_gauss=fit_param_restrict_dict_nl_gauss,
            fit_param_restrict_dict_nl_lorentz=fit_param_restrict_dict_nl_lorentz,
            fit_param_restrict_dict_bl_gauss=fit_param_restrict_dict_bl_gauss)

        minimizer = iminuit.Minuit(self.calc_model_predict_chi2_val, *init_guess_list, name=param_name_list)
        # set restrictions
        self.set_param_restrict(minimizer=minimizer,
                                fit_param_restrict_dict_nl_gauss=fit_param_restrict_dict_nl_gauss,
                                fit_param_restrict_dict_nl_lorentz=fit_param_restrict_dict_nl_lorentz,
                                fit_param_restrict_dict_bl_gauss=fit_param_restrict_dict_bl_gauss)

        minimizer.migrad()
        print(minimizer)
        minimizer.hesse()
        print(minimizer)

        best_fit = self.calc_model_predict(*minimizer.values)
        chi2 = self.calc_model_predict_chi2_val(*minimizer.values)
        ndof = sum(np.invert(np.isnan(self.x_data))) - len(init_guess_list)

        fit_result_dict = {'best_fit': best_fit, 'chi2': chi2, 'ndof': ndof}
        fit_result_dict = self.get_fit_result_dict(minimizer=minimizer, fit_result_dict=fit_result_dict)

        return fit_result_dict





