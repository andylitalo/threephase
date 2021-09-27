# -*- coding: utf-8 -*-
"""
Created end of September 2019

This document defines the Experiment class along with the subclass Condition
and its subclass Measurement. This class structure is intended to organize the
measurements taken by various readouts from the ternary sampling instrument
designed and assembled in the shared walk-in hood of Modules 60-61 in the ECB
building at Dow's Lake Jackson research facility.

@author: Andy
"""

import numpy as np
import pandas as pd
import pickle as pkl

from timedate import TimeDate
import tern

# GLOBAL CONSTANTS
P_ATM = 14.7 # atmospheric pressure [psi]
T_ATM = 21 # atmospheric temperature (in the lab) [C]


class Experiment:
    """
    An object in the Experiment class (an "experiment") is  a collection of
    pressure steps undergone for a single recipe of cyclopentane and polyol
    loaded in the Parr reactor.
    """
    def __init__(self, m_poly, m_c5, m_impurities, V, p_isco, T_isco, V_isco,
                timedate_0, mixing_rate, V_per_meas=2, p_parr_offset=10,
                p_tank=851, V_isco_full=508, m_co2=0):
        """
        Creates new Experiment object.
        PARAMETERS:
            self : object
                The created Experiment object.
            m_poly : float
                Mass of polyol dispensed into the Parr reactor [g].
            m_c5 : float
                Mass of cyclopentane dispensed into the Parr reactor [g].
            m_impurities : float
                Mass of impurities dispensed into the Parr reactor [g].
            V : float
                Volume of the Parr reactor [mL].
            p_isco : float
                Pressure in the ISCO pump at the current stage in the experiment [psi].
            T_isco : float
                Temperature of the isco pump at the current stage of the experiment [C].
            V_isco : float
                Volume of the syringe in the ISCO pump at the current stage of the experiment [mL].
            timedate_0 : TimeDate object (from timedate.py)
                Date and time of beginning of the experiment.
            mixing_rate : float
                Mixing rate of the Parr reactor [RPM].
            V_per_meas : float, default=2
                Volume extracted per liquid measurement [mL].
            p_parr_offset : float, default=10
                Offset of pressure in Parr (i.e. p_true = p_parr - p_parr_offset) [psi].
                Parr read 25 psi at atmosphere, which is 14.7 psi --> offset ~ 10 psi
                (for absolute pressure, which is what is most relevant to)
            p_tank : float, default=851 (condensation pressure of CO2 at room temperature)
                Pressure in liquid CO2 tank [psi].
            V_isco_full : float, default=508 (value for ISCO 500DM)
                Volume in ISCO pump when full
        """
        # Store experimental parameters
        self.m_poly_0 = m_poly # initial mass of polyol in Parr reactor [g]
        self.m_poly = m_poly # current mass of polyol in Parr reactor [g]
        self.m_c5_0 = m_c5 # initial mass of cyclopentane in Parr reactor [g]
        self.m_c5 = m_c5 # current mass of cyclopentane in Parr reactor [g]
        self.m_impurities = m_impurities # mass of impurities in Parr reactor [g]
        self.V = V # volume of the Parr reactor [mL]
        self.p_isco = p_isco # current pressure in the ISCO pump [psi]
        self.T_isco = T_isco # current temperature of the ISCO pump [C] (ambient)
        self.V_isco = V_isco # current volume in the cylinder of the ISCO pump [mL]
        self.timedate_0 = timedate_0 # TimeDate object representing start of experiment
        self.V_per_meas = V_per_meas # approximate liquid volume sampled per measurement [mL]
        self.p_parr_offset = p_parr_offset # offset on the Parr reactor's pressure gauge [psi]
        self.mixing_rate = mixing_rate # mixing rate in RPM
        self.p_tank = p_tank # pressure of CO2 tank [psi]
        self.V_isco_full = V_isco_full # volume of ISCO pump cylinder when full [mL]
        # compute initial amount of CO2 in ISCO pump [g]
        self.m_co2_isco_0 = tern.rho_co2(p_isco, T_isco, psi=True)*V_isco
        # initialize list of pressure steps in experiment
        self.cond_list = []
        # set current amount of CO2 in ISCO pump to initial amount
        self.m_co2_isco = self.m_co2_isco_0
        # initialize mass of CO2 in Parr reactor [g]
        self.m_co2 = m_co2
        # initialize mass of CO2 that leaked out of ISCO pump [g]
        self.m_co2_isco_leak = 0
        # initialize mass of CO2 refilled into ISCO from CO2 tank [g]
        self.m_co2_refill = 0

    def load_cond(self, cond):
        """Loads Condition object cond to list cond_list in Experiment object."""
        # Only load Condition object if not already present
        if cond not in self.cond_list:
            self.cond_list += [cond]

    def get_p(self):
        """Returns pressures of each experiment in chronological order."""
        return np.array([m.p for cond in self.cond_list for m in cond.measurement_list])

    def get_T(self):
        """Returns pressures of each experiment in chronological order."""
        return np.array([m.T for cond in self.cond_list for m in cond.measurement_list])

    def get_meas(self, p_min=0, p_max=1500, T_min=-247, T_max=60):
        """Returns measurements within limits of p [psi] and T [C] given."""
        return [m for cond in self.cond_list for m in cond.measurement_list
                if m.p >= p_min and m.p <= p_max and m.T >= T_min and m.T <= T_max]

    def get_m_co2_leaked(self, leak_rate_list):
        """Returns mass of co2 leaked at each measurement based on leak rates given [g/min]."""
        # initialize total mass of leaked CO2 at previous pressure step [g]
        m_co2_leak_prev = 0
        # initialize list of mass of leaked CO2 for each measurement [g]
        m_co2_leak_list = []
        # initialize counter of pressure steps
        i = 0
        # loop through pressure steps
        for cond in self.cond_list:
            # get estimated leak rate of CO2 for current pressure step [g/min]
            leak_rate = leak_rate_list[i]
            # loop through each measurement to estimate leaked CO2 [g]
            for m in cond.measurement_list:
                # get time since start of pressure step [min]
                dt = TimeDate.diff_min(cond.timedate, m.timedate)
                # calculate mass of leaked CO2 by adding leak during current
                # pressure step to total leak at end of previous pressure step [g]
                m_co2_leak_list += [leak_rate*dt+m_co2_leak_prev]
            # update mass of CO2 leaked at end of pressure step
            m_co2_leak_prev = m_co2_leak_list[-1]
            # update counter of pressure steps
            i += 1

        return m_co2_leak_list

    def to_df(self, m_co2_leak_list=[]):
        """Converts experiment into a dataframe."""
        # get list of measurements
        m_list = self.get_meas()
        # counter measurements
        n = len(m_list)
        # initialize dataframe with column names in order
        df = pd.DataFrame(columns=['date', 'time', 'elapsed time [min]',
                        'pressure [psi]', 'temperature [C]',
                        'mass co2 [g]', 'mass co2 (leak) [g]', 'mass co2 [g] (corr)',
                        'mass c5 [g]', 'mass polyol [g]',
                        'pressure (isco) [psi]', 'temperature (isco) [C]',
                        'volume (isco) [mL]', 'mass co2 (isco) [g]',
                        'mass co2 leak (isco) [g]', 'mass co2 (refill) [g]',
                        'peak area (back) co2 [a.u.]', 'peak area (back) c5 [a.u.]',
                        'peak area (front) co2 [a.u.]', 'peak area (front) c5 [a.u.]',
                        'volume sampled (vap) [mL]',
                        'density co2 (vap) [g/mL]', 'density c5 (vap) [g/mL]',
                        'density co2 (liq) [g/mL]', 'density c5 (liq) [g/mL]',
                        'density polyol (liq) [g/mL]', 'density co2 (tot) [g/mL]',
                        'density co2 (tot) [g/mL] (corr)',
                        'density c5 (tot) [g/mL]', 'density polyol (tot) [g/mL]',
                        'w co2 (vap) [w/w]', 'w c5 (vap) [w/w]', 'w co2 (liq) [w/w]',
                        'w c5 (liq) [w/w]', 'w polyol (liq) [w/w]',
                        'w co2 (tot) [w/w]', 'w c5 (tot) [w/w]',
                        'w polyol (tot) [w/w]', 'w co2 (tot) [w/w] (corr)', 'w c5 (tot) [w/w] (corr)',
                        'w polyol (tot) [w/w] (corr)', 'liquid volume [mL]', 'mass c5 (pred) [g]',
                        'mass c5 (missing) [g]',  'mass co2 (pred) [g]',
                        'mass co2 (missing) [g]', 'is error'],
                        index=range(n) )
        # loop through each measurement and assign values to dataframe columns
        for i in range(n):
            # get measurement
            m = m_list[i]
            # compute total mass of material in Parr reactor [g]
            m_tot = m.m_co2 + m.m_c5 + m.m_poly + m.cond.experiment.m_impurities
            df['date'].iloc[i] = m.timedate.get_date_string()
            df['time'].iloc[i] = m.timedate.get_time_string()
            df['elapsed time [min]'].iloc[i] = m.elapsed_time
            df['mass co2 [g]'].iloc[i] = m.m_co2
            df['mass c5 [g]'].iloc[i] = m.m_c5
            df['mass polyol [g]'].iloc[i] = m.m_poly
            df['pressure (isco) [psi]'].iloc[i] = m.p_isco
            df['temperature (isco) [C]'].iloc[i] = m.T_isco
            df['volume (isco) [mL]'].iloc[i] = m.V_isco
            df['mass co2 (isco) [g]'].iloc[i] = m.m_co2_isco
            df['mass co2 leak (isco) [g]'].iloc[i] = m.m_co2_isco_leak
            df['mass co2 (refill) [g]'].iloc[i] = m.m_co2_refill
            df['peak area (back) co2 [a.u.]'].iloc[i] = m.pa_b_co2
            df['peak area (back) c5 [a.u.]'].iloc[i] = m.pa_b_c5
            df['peak area (front) co2 [a.u.]'].iloc[i] = m.pa_f_co2
            df['peak area (front) c5 [a.u.]'].iloc[i] = m.pa_f_c5
            df['pressure [psi]'].iloc[i] = m.p
            df['temperature [C]'].iloc[i] = m.T
            df['volume sampled (vap) [mL]'].iloc[i] = m.V_vap_sampled
            df['density co2 (vap) [g/mL]'].iloc[i] = m.rho_v_co2
            df['density c5 (vap) [g/mL]'].iloc[i] = m.rho_v_c5
            df['density co2 (liq) [g/mL]'].iloc[i] = m.rho_l_co2
            df['density c5 (liq) [g/mL]'].iloc[i] = m.rho_l_c5
            df['density polyol (liq) [g/mL]'].iloc[i] = m.rho_l_poly
            df['density co2 (tot) [g/mL]'].iloc[i] = m.m_co2 / m.cond.experiment.V ###
            df['density c5 (tot) [g/mL]'].iloc[i] = m.m_c5 / m.cond.experiment.V
            df['density polyol (tot) [g/mL]'].iloc[i] = m.m_poly / m.cond.experiment.V
            df['w co2 (vap) [w/w]'].iloc[i] = m.w_v_co2
            df['w c5 (vap) [w/w]'].iloc[i] = m.w_v_c5
            df['w co2 (liq) [w/w]'].iloc[i] = m.w_l_co2
            df['w c5 (liq) [w/w]'].iloc[i] = m.w_l_c5
            df['w polyol (liq) [w/w]'].iloc[i] = m.w_l_poly
            df['w co2 (tot) [w/w]'].iloc[i] = m.m_co2/(m_tot) ###
            df['w c5 (tot) [w/w]'].iloc[i] = m.m_c5/(m_tot)
            df['w polyol (tot) [w/w]'].iloc[i] = m.m_poly/(m_tot)
            df['liquid volume [mL]'].iloc[i] = m.V_liq
            df['mass c5 (pred) [g]'].iloc[i] = m.m_c5_pred
            df['mass c5 (missing) [g]'].iloc[i] = m.m_c5_missing
            df['mass co2 (pred) [g]'].iloc[i] = m.m_co2_pred
            df['mass co2 (missing) [g]'].iloc[i] = m.m_co2_missing
            df['is error'].iloc[i] = m.is_error
            # if mass of CO2 leak is provided, also compute corrected values
            if len(m_co2_leak_list) > 0:
                # get mass of CO2 leak [g]
                m_co2_leak = m_co2_leak_list[i]
                # correct mass of CO2 in Parr by subtracting leaked mass [g]
                m_co2_corr = m.m_co2 - m_co2_leak
                # similarly, correct total mass of material [g]
                m_tot_corr = m_co2_corr + m.m_c5 + m.m_poly + m.cond.experiment.m_impurities
                df['mass co2 [g] (corr)'].iloc[i] = m_co2_corr
                df['mass co2 (leak) [g]'].iloc[i] = m_co2_leak
                df['density co2 (tot) [g/mL] (corr)'].iloc[i] = m_co2_corr / m.cond.experiment.V ###
                df['w co2 (tot) [w/w] (corr)'].iloc[i] = m_co2_corr/(m_tot_corr) ###
                df['w c5 (tot) [w/w] (corr)'].iloc[i] = m.m_c5/(m_tot_corr)
                df['w polyol (tot) [w/w] (corr)'].iloc[i] = m.m_poly/(m_tot_corr)

        return df

    def save_metadata(self, file_hdr='metadata'):
        """
        Saves metadata in python dictionary.
        ***LIKELY NON-OPERATIVE BECAUSE PYTHON WILL NOT PICKLE EXPERIMENT OBJECTS.
        """
        metadata = {'volume [mL]':self.V, 'mass impurities [g]':self.m_impurities,
                    'start mass of co2 (isco) [g]':self.m_co2_isco_0,
                    'start time':self.timedate_0.get_time_string(),
                    'start date':self.timedate_0.get_date_string(),
                    'volume per measurement [mL]':self.V_per_meas,
                    'offset of Parr pressure [psi]':self.p_parr_offset}
        with open(file_hdr + '_{0:d}'.format(100*self.m_c5_0) + '%c5.pkl', 'wb') as f:
            pkl.dump(metadata, f)

class Condition(Experiment):
    """
    A collection of "measurements" (objects of the Measurement class) taken
    when the Parr reactor was under a given set of conditions. These conditions
    can be changed by 1) adding CO2 with the ISCO pump, 2) adding C5 with the
    ISCO pump, or 3) venting the headspace through the gas-sampling port.
    Each change in condition is handled differently.
    """

    def adjust_p_isco(self, experiment, params):
        """
        """
        p_isco_i, p_isco_f, V_isco_i, V_isco_f, p_isco_refill = params
        # use offset of ISCO pump's pressure transducer from previous condition
        if len(experiment.cond_list) > 1:
            self.p_isco_offset = experiment.cond_list[-2].p_isco_offset
        # otherwise set as default to 0
        else:
            self.p_isco_offset = 0

        # store ISCO data for case where ISCO refilled before pressure step
        if p_isco_refill > 0:
            # use refill pressure to determine offset (should match tank pressure)
            self.p_isco_offset = p_isco_refill - experiment.p_tank
            # compute masses of CO2 before and after refill [g]
            m_co2_i = tern.rho_co2(p_isco_i-self.p_isco_offset, experiment.T_isco, psi=True)*V_isco_i
            m_co2_full = tern.rho_co2(experiment.p_tank, experiment.T_isco, psi=True)*experiment.V_isco_full
            # compute difference to get amount of CO2 extracted [g]
            experiment.m_co2_refill += m_co2_full - m_co2_i
            # reset beginning pressure in ISCO to refilled conditions
            p_isco_i = experiment.p_tank
            V_isco_i = experiment.V_isco_full
        # store data for case where ISCO was refilled after pressure step
        elif p_isco_refill < 0:
            # use refill pressure to determine offset (should match tank pressure)
            self.p_isco_offset = (-p_isco_refill) - experiment.p_tank
            # compute how much co2 was in ISCO before refilling
            m_co2_f = tern.rho_co2(p_isco_f-self.p_isco_offset, experiment.T_isco, psi=True)*V_isco_f
            # compute how much co2 was in isco once refilled
            m_co2_full = tern.rho_co2(experiment.p_tank, experiment.T_isco, psi=True)*experiment.V_isco_full
            # subtract full tank from beginning to get refilled CO2 amount
            experiment.m_co2_refill += m_co2_full - m_co2_f
        # store corrected values of ISCO
        p_isco_i -= self.p_isco_offset
        p_isco_f -= self.p_isco_offset

        return p_isco_i, p_isco_f

    def add_co2(self, experiment, params):
        """
        """
        p_isco_i, p_isco_f = self.adjust_p_isco(experiment, params)
        _, _, V_isco_i, V_isco_f, p_isco_refill = params
        # compute masses of co2 before and after pressurization  [g]
        m_co2_i = tern.rho_co2(self.p_isco_i, experiment.T_isco, psi=True)*V_isco_i
        m_co2_f = tern.rho_co2(self.p_isco_f, experiment.T_isco, psi=True)*V_isco_f
        self.m_co2_isco = m_co2_f
        # add co2 lost from isco to dispensed co2 in Parr
        experiment.m_co2 += m_co2_i - m_co2_f

    def add_c5(self, experiment, params):
        """
        """
        p_isco_i, p_isco_f = self.adjust_p_isco(experiment, params)
        _, _, V_isco_i, V_isco_f, p_isco_refill = params
        m_c5_i = tern.rho_c5(experiment.T_isco)*V_isco_i
        m_c5_f = tern.rho_c5(experiment.T_isco)*V_isco_f
        self.m_co2_isco = 0
        experiment.m_c5 += m_c5_i - m_c5_f

    def vent(self, experiment, params):
        """
        """
        

    def __init__(self, experiment, p0, T0, timedate, cond_change, params=[]):
        """
        Creates Condition object initialized with the given parameters.
        PARAMETERS:
            self : Condition object
                Object created in this method.
            experiment : Experiment object
                Experiment to which this pressure step belongs.
            p0 : float
                Initial pressure reading on the Parr reactor after pressure step [psi].
            T0 : float
                Initial temperature reading from the Parr reactor after pressure step [C].
            p_isco_i : float
                Pressure in the ISCO pump before pressurizing the Parr reactor [psi].
            V_isco_i : float
                Volume in the ISCO pump before pressurizing the Parr reactor [mL].
            p_isco_f : float
                Pressure in the ISCO pump after pressurizing the Parr reactor [psi].
            V_isco_f : float
                Volume in the ISCO pump after pressurizing the Parr reactor [mL].
            timedate : TimeDate object (see timedate.py)
                Date and time of pressurization.
            p_isco_refill : float, default=-1
                Pressure in ISCO pump after refill [psi]. If ISCO not refilled during
                this pressure step, leave as the default of -1.
            T_isco : float, default=-274
                Temperature of ISCO pump [C]. If same as throughout rest of
                experiment, leave as default of -274.
            c5 : bool, default=False
                If True, indicates that ISCO pump has cyclopentane.
        """
        # store initial pressure and temperature in Parr reactor
        self.p0 = p0-experiment.p_parr_offset # [psi]
        self.T0 = T0 # [C]
        # store experient in order to access its parameters
        self.experiment = experiment
        # upload pressure step to experiment it comes from if not already there
        experiment.load_cond(self)
        # initialize list of measurements included in this pressure step
        self.measurement_list = []
        # record time and date of pressurization
        self.timedate = timedate

        if cond_change=='add co2':
            self.add_co2(experiment, params)
        elif cond_change=='add c5':
            self.add_c5(experiment, params)
        elif cond_change=='vent':
            self.vent(experiment, params)
        else:
            assert True, "Please select a valid option."

        # store co2 masses
        self.m_co2 = experiment.m_co2
        self.m_c5 = experiment.m_c5
        self.m_poly = experiment.m_poly
        self.m_co2_isco_leak = experiment.m_co2_isco_leak
        self.m_co2_refill = experiment.m_co2_refill
        experiment.m_co2_isco = self.m_co2_isco

        # equilibrium measurement index (default is last, can be changed later)
        self.ind_equil = -1



    def load_measurement(self, measurement, overwrite=False):
        """Loads Measurement object to list in Experiment object."""
        # Only load measurement if not already present
        if measurement not in self.measurement_list:
            self.measurement_list += [measurement]
        # overwrite the previous measurement
        elif overwrite:
            ind = [i for i in range(len(measurement_list)) if measurement==measurement_list[i]]
            measurement_list[ind] = measurement

    def get_average_pa(self):
        """Averages peak areas of measurements, excluding erroneous ones."""
        result = []
        pa_list = [pa_b_c5, pa_b_co2, pa_f_c5, pa_f_co2]
        ct = 0
        # loop through back and front signals, c5 and co2 compounds
        for signal in ['b', 'f']:
            for compound in ['c5', 'co2']:
                # get list of values for all non-error measurements
                to_average = [tern.pa_conv(pa_list[ct], signal, compound)
                            for m in self.measurement_list if not m.is_error]
                # average values and add to result
                result += [np.mean(np.array(to_average))]
                ct += 1

        return result

    def get_equil_meas(self):
        """Returns measurement designated as the equilibrium measurement."""
        return self.measurement_list[self.ind_equil]

    def get_leak_info(self):
        """Returns data required to estimate the leak rate of vapor phase."""
        # get final measurement
        m_f = self.measurement_list[-1]

        return np.array([self.p0, self.T0, self.m_co2, self.m_poly, self.m_c5,
                    m_f.p, m_f.T, m_f.m_co2, m_f.m_poly, m_f.m_c5,
                    TimeDate.diff_min(self.timedate, m_f.timedate)])


class Measurement(Condition):
    """
    A "measurement" includes the data of a single sample from the GC and the
    status measurements (p, T, etc.) of the equipment at the time.
    """
    def compute_rho(self):
        """Computes densities of each component in each phase."""
        rho_poly = tern.rho_v2110b(self.T)
        rho_c5 = tern.rho_c5(self.T)
        # this estimation of polyol and CO2 densities based on Naples data
        # results in a lower estimation of liquid volume and thus greater
        # missing C5 mass
        rho_co2 = rho_poly
        # convert peak areas to densities [g/mL]
        self.rho_l_c5 = tern.pa_conv(self.pa_f_c5, 'f', 'c5')
        self.rho_v_c5 = tern.pa_conv(self.pa_b_c5, 'b', 'c5')
        self.rho_l_co2 = tern.pa_conv(self.pa_f_co2, 'f', 'co2')
        self.rho_v_co2 = tern.pa_conv(self.pa_b_co2, 'b', 'co2')
        # estimate density of polyol in the liquid phase sampled by HPLIS [g/mL]
        self.rho_l_poly = rho_poly*(1 - (self.rho_l_co2/rho_co2) - (self.rho_l_c5/rho_c5))

    def compute_wt_frac(self):
        """Computes and stores weight fractions of each component in each phase."""
        # compute vapor-phase densities
        rho_v = self.rho_v_co2 + self.rho_v_c5
        self.w_v_co2 = self.rho_v_co2 / rho_v
        self.w_v_c5 = self.rho_v_c5 / rho_v
        # compute liquid-phase densities
        rho_l = self.rho_l_co2 + self.rho_l_c5 + self.rho_l_poly
        self.w_l_co2 = self.rho_l_co2 / rho_l
        self.w_l_c5 = self.rho_l_c5 / rho_l
        self.w_l_poly = self.rho_l_poly / rho_l

    def est_V_Liq(self):
        """Estimates the volume of the liquid (densest) phase."""
        # estimate the volume of the liquid phase
        self.V_liq = self.m_poly / self.rho_l_poly
        # the following formula vastly (up to factor of 2) overestimates the
        # liquid-phase volume and thus also the mass of c5
        # tmp = self.V_liq
        # self.V_liq = (self.m_poly/tern.rho_v2110b(self.T) + (self.m_c5-self.rho_v_c5*self.experiment.V)/ \
        #             tern.rho_c5(self.T) + (self.m_co2-self.rho_v_co2*self.experiment.V)/ \
        #             tern.rho_v2110b(self.T))/(1-self.rho_v_c5/tern.rho_c5(self.T) \
        #             -self.rho_v_co2/tern.rho_v2110b(self.T))
        # print(tmp, self.V_liq)

    def pred_m_c5(self):
        """Computes missing mass of cyclopentane as sign of 3-phase region"""
        # estimate cyclopentane mass by adding up masses in liquid and vapor phases
        self.m_c5_pred = self.V_liq*self.rho_l_c5 + (self.cond.experiment.V-self.V_liq)*self.rho_v_c5

    def pred_m_co2(self):
        """Computes missing mass of co2 as sign of 3-phase region"""
        # estimate cyclopentane mass by adding up masses in liquid and vapor phases
        self.m_co2_pred = self.V_liq*self.rho_l_co2 + (self.cond.experiment.V-self.V_liq)*self.rho_v_co2

    def correct_rho_vap(self, filename='co2_c5_vle.csv', thresh=0.4):
        """
        Corrects density measured with GC for vapor phase by using a PC-SAFT
        model fitted to experimental measurements of the CO2-C5 VLE from
        Eckert and Sandler (1986) to estimate the expected total density of the
        vapor phase of the mixture, then scales the GC measurements to have the
        same total density. Essentially, the weight fractions are kept from the
        GC measurements, but the overall density is estimated with a PC-SAFT
        model fitted to empirical data.
        """
        rho_tot = np.sum(tern.rho_v_co2_c5(self.p, self.T, self.w_v_co2, self.w_v_c5,
                                            filename=filename, thresh=thresh))
        rho_gc = self.rho_v_co2 + self.rho_v_c5
        # correct GC measurement
        correction = rho_tot/rho_gc
        self.rho_v_co2 *= correction
        self.rho_v_c5 *= correction


    def __init__(self, cond, pa_f_co2, pa_f_c5, pa_b_co2, pa_b_c5, p_parr, T,
                V_isco, V_vap_sampled_atm, timedate, p_isco=-1, T_isco=-274,
                mixing_stopped=True, is_error=False, sampled_liquid=True):
        """
        Creates a Measurement object with the given parameters.
        PARAMETERS:
            self : Measurement object
                Object created by this method.
            cond : Condition object
                Pressure step to which this measurement belongs.
            pa_f_co2 : float
                Peak area of CO2 peak from the front GC signal (liquid).
            pa_f_c5 : float
                Peak area of C5 peak from the front GC signal (liquid).
            pa_b_co2 : float
                Peak area of CO2 peak from the back GC signal (vapor).
            pa_b_c5 : float
                Peak area of C5 peak from the back GC signal (vapor).
            p_parr : float
                Pressure of Parr reactor during measurement [psi].
            T : float
                Temperature of Parr reactor during measurement [C].
            V_isco : float
                Volume in the ISCO pump at beginning of measurement [mL]. Used
                to determine how much CO2 leaked out of the ISCO between
                measurements.
            V_vap_sampled_atm : float
                Volume of vapor phase sampled under atmospheric pressure and
                temperature (estimated using flow meter) [mL].
            timedate : TimeDate object (see timedate.py)
                Date and time of measurement according to GC computer.
            p_isco : float, default=-1
                Pressure of ISCO pump [psi]. Assumed same as start of experiment
                if left at default of -1.
            T_isco : float, default=-274:
                Temperature of ISCO pump [C]. Assumed same as start of experiment
                if left at default of -274.
            mixing_stopped : bool, default=True
        """
        # if any peak areas are negative, declare measurement as an error
        if np.any(np.array([pa_f_co2, pa_f_c5, pa_b_co2, pa_b_c5]) < 0):
            is_error = True
        # load experiment corresponding to the pressure step for easy access
        experiment = cond.experiment
        # store measurement data
        self.pa_f_co2 = pa_f_co2
        self.pa_f_c5 = pa_f_c5
        self.pa_b_co2 = pa_b_co2
        self.pa_b_c5 = pa_b_c5
        # store non-GC measurements
        self.p = p_parr - experiment.p_parr_offset
        self.T = T
        self.timedate = timedate
        # compute elapsed time since start of experiment [min]
        self.elapsed_time = TimeDate.diff_min(experiment.timedate_0,
                                                timedate)
        # record the time that things have been diffusing
        self.diffusion_time = TimeDate.diff_min(cond.timedate, timedate)
        # record whether mixing was stopped during experiment or not
        self.mixing_stopped = mixing_stopped
        # store pressure step in which this measurement is taken
        self.cond = cond
        # add measurement to experiment's measurement list
        cond.load_measurement(self)
        # store ISCO data
        if p_isco == -1:
            p_isco = experiment.p_isco - self.cond.p_isco_offset
        if T_isco == -274:
            T_isco = experiment.T_isco
        self.p_isco = p_isco - self.cond.p_isco_offset
        self.T_isco = T_isco
        self.V_isco = V_isco
        self.m_co2_isco = tern.rho_co2(self.p_isco, self.T_isco, psi=True)*self.V_isco
        d_co2 = experiment.m_co2_isco - self.m_co2_isco
        if d_co2 > 0:
            experiment.m_co2_isco_leak += d_co2
        else:
            experiment.m_co2_refill -= d_co2
        self.m_co2_isco_leak = experiment.m_co2_isco_leak
        self.m_co2_refill = experiment.m_co2_refill
        experiment.m_co2_isco = self.m_co2_isco
        self.is_error = is_error
        self.sampled_liquid = sampled_liquid
        # compute and store densities of each phase
        self.compute_rho()
        # compute and store weight fractions
        self.compute_wt_frac()
        # correct mass of each component to account for liquid lost to sampling
        if sampled_liquid:
            experiment.m_c5 -= self.rho_l_c5*experiment.V_per_meas
            experiment.m_poly -= self.rho_l_poly*experiment.V_per_meas
            experiment.m_co2 -= self.rho_l_co2*experiment.V_per_meas
        # correct mass of co2 and c5 to account for loss during gas sampling
        rho_atm = np.sum(tern.rho_v_co2_c5(P_ATM, T_ATM, self.w_v_co2, self.w_v_c5))
        rho_parr = np.sum(tern.rho_v_co2_c5(self.p, self.T, self.w_v_co2, self.w_v_c5))
        self.V_vap_sampled = (rho_atm/rho_parr) * V_vap_sampled_atm
        experiment.m_co2 -= self.rho_v_co2*self.V_vap_sampled
        experiment.m_c5 -= self.rho_v_c5*self.V_vap_sampled
        # record current masses of components in measurement object [g]
        self.m_c5 = experiment.m_c5
        self.m_poly = experiment.m_poly
        self.m_co2 = experiment.m_co2
        # compute volume of liquid phase [mL]
        self.est_V_Liq()
        # correct densities of vapor phase
        self.correct_rho_vap()
        # predict mass of cyclopentane [g]
        self.pred_m_c5()
        # compute missing mass of cyclopentane as sign of 3-phase region [g]
        self.m_c5_missing = self.m_c5 - self.m_c5_pred
        # predict mass of co2 [g]
        self.pred_m_co2()
        # compute missing mass of co2 as a sign of the 3-phase region [g]
        self.m_co2_missing = self.m_co2 - self.m_co2_pred

    # def get_m_co2(self, timedate_leak=TimeDate(date_str='10/03/2019', time_str='9:30:00'),
    #                 rate_leak=0.03625):
    #     """Returns mass of CO2 corrected by estimated linear leak rate."""
    #     time_leak = TimeDate.diff_min(timedate_leak, self.timedate)
    #     m_co2_isco_leaked = time_leak*rate_leak*(time_leak>0)
    #
    #     return self.m_co2 - m_co2_isco_leaked







def __init__():
    pass
