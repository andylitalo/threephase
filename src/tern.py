"""
tern is a library of methods used for analyzing data from GC sampling of ternary
systems composed of cyclopentane, polyol, and CO2.

Author - Andy Ylitalo
Created 16:20, 10/2/19, CDT
"""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from timedate import TimeDate

# GLOBAL CONSTANTS
P_ATM = 14.7 # atmospheric pressure [psi]
T_ATM = 21 # atmospheric temperature (in the lab) [C]


def analyze_data(df_gc, df_cond, m_poly_i, m_c5_i, V_liq_i, T_isco,
                    timedate_start, p_parr_offset, V_liq_sampled, V):
    """
    Analyzes data from gas chromatography (GC) sampling of a ternary CO2-C5-polyol
    mixtures in a Parr reactor and loaded with an ISCO pump. The experiments
    were performed from October 2nd to October 11th, 2019 at ECB building at the
    Lake Jackson, TX research site of Dow. Raw data for the GC measurements and
    the conditions changes (injections from the ISCO pump or venting of the
    Parr reactor) are loaded and used to make estimates of the compositions of
    the various phases. The amount of C5 and CO2 not accounted for by the GC
    measurements is considered "missing" and assumed to be condensed in an
    intermediate third phase between the polyol-rich liquid phase and the CO2-
    rich vapor phase, although this phase has not been demonstrated rigorously.

    PARAMETERS:
        df_gc : Pandas dataframe
            Dataframe of raw GC data.
        df_cond : Pandas dataframe
            Dataframe of raw data for changes in the condition (injections of
            CO2 or C5, or venting from the gas-sampling port of the Parr reactor).
        m_poly_i : float
            Initial mass of polyol in the Parr reactor [g].
        m_c5_i : float
            Initial mass of cyclopentane in the Parr reactor [g].
        V_liq_i : float
            Initial volume of liquid in the Parr reactor [mL].
        T_isco : float
            Temperature of ISCO pump (assumed to be lab temperature) [C].
        timedate_start : TimeDate object (see timedate.py)
            Time and date of start of experiment.
        p_parr_offset : float
            Offset of Parr reactor's pressure gauge above true pressure [psi].
        V_liq_sampled : numpy array of floats
            Array of volume of liquid sampled at each GC measurement [mL].
        V : float
            Internal volume of Parr reactor [mL].

    RETURNS:
        df : Pandas dataframe
            Dataframe of all processed data, along with relevant raw data.
    """
    # initialize dataframe to store data
    df = pd.DataFrame()

    # extract data from dataframe
    date_str = [timedate_start.get_date_string()] + list(df_gc['date'])
    time_str = [timedate_start.get_time_string()] + list(df_gc['time'])
    pa_f_co2 = [0] + list(df_gc['peak area co2 (front) [a.u.]'])
    pa_f_c5 = [0] + list(df_gc['peak area c5 (front) [a.u.]'])
    pa_b_co2 = [0] + list(df_gc['peak area co2 (back) [a.u.]'])
    pa_b_c5 = [0] + list(df_gc['peak area c5 (back) [a.u.]'])
    p_parr = [0] + list(df_gc['pressure (parr) [psi]'].to_numpy(dtype=float) - p_parr_offset)
    T_parr = [T_isco] + list(df_gc['temperature (parr) [C]'])
    V_vap_sampled_atm = [0] + list(df_gc['volume sampled (vap) [mL] (atm)'])
    V_liq_sampled = [0] + list(V_liq_sampled)
    # create timedate list from date and times of measurements
    timedate = [TimeDate(date_str=date_str[i], time_str=time_str[i]) for i in range(len(date_str))]

    # extract relevant data from condition change dataframe
    timedate_cc = [TimeDate(date_str=df_cond['date'].iloc[i], \
                    time_str=df_cond['time'].iloc[i]) for i in \
                    range(len(df_cond))]
    elapsed_time_cc = np.array([TimeDate.diff_min(timedate_start, td_cc) \
                                for td_cc in timedate_cc])
    # some intitial calculations
    m_i = m_poly_i + m_c5_i
    rho_i = m_i/V_liq_i
    w_c5_i = m_c5_i / m_i
    w_poly_i = 1 - w_c5_i
    rho_l_c5_i = w_c5_i*rho_i
    rho_l_poly_i = w_poly_i*rho_i

    # initialize values to be calculated
    m_co2 = [0]
    m_co2_bin = [0]
    m_poly = [m_poly_i]
    m_c5 = [m_c5_i]
    elapsed_time = [0] #minutes
    V_liq = [V_liq_i]
    V_vap_sampled = [0]
    rho_v_co2 = [0]
    rho_v_c5 = [0]
    rho_l_co2 = [0]
    rho_l_c5 = [rho_l_c5_i]
    rho_l_poly = [rho_l_poly_i]
    rho_tot_co2 = [0]
    rho_tot_c5 = [m_c5_i/V]
    rho_tot_poly = [m_poly_i/V]
    w_v_co2 = [0]
    w_v_c5 = [0]
    w_l_co2 = [0]
    w_l_c5 = [w_c5_i]
    w_l_poly = [w_poly_i]
    w_co2 = [0]
    w_c5 = [w_c5_i]
    w_poly = [w_poly_i]
    m_c5_pred = [m_c5_i]
    m_c5_missing = [0]
    m_co2_pred = [0]
    m_co2_missing = [0]
    dm_co2 = [0]
    dm_co2_bin = [0]
    dm_poly = [0]
    dm_c5 = [0]

    # initialize counter for condition change
    j_prev = -1
    # analysis by looping through GC data and appending processed results to lists
    for i in range(1, len(pa_f_co2)):
        # compute densities measured by GC [g/mL]
        rho_l_co2 += [pa_conv(pa_f_co2[i], 'f', 'co2')]
        rho_l_c5 += [pa_conv(pa_f_c5[i], 'f', 'c5')]
        rho_v_co2_gc = pa_conv(pa_b_co2[i], 'b', 'co2')
        rho_v_c5_gc = pa_conv(pa_b_c5[i], 'b', 'c5')
        rho_v_co2_new, rho_v_c5_new = correct_rho_vap(p_parr[i], T_parr[i], rho_v_co2_gc, rho_v_c5_gc)
        rho_v_co2 += [rho_v_co2_new]
        rho_v_c5 += [rho_v_c5_new]
        # compute density of polyol in liquid phase [g/mL]
        rho_poly_co2 = rho_polyol_co2(p_parr[i], T_parr[i])
        rho_l_poly += [rho_poly_co2*(1 - (rho_l_co2[i]/rho_poly_co2) - \
                        (rho_l_c5[i]/rho_c5(T_parr[i])))]
        # compute weight fractions based on GC measurements [w/w]
        rho_l = rho_l_co2[i] + rho_l_poly[i] + rho_l_c5[i]
        w_l_co2 += [rho_l_co2[i]/rho_l]
        w_l_poly += [rho_l_poly[i]/rho_l]
        w_l_c5 += [rho_l_c5[i]/rho_l]
        rho_v = rho_v_co2[i] + rho_v_c5[i]
        w_v_co2 += [rho_v_co2[i]/rho_v]
        w_v_c5 += [rho_v_c5[i]/rho_v]

        # update mass of polyol
        dm_poly_curr = -V_liq_sampled[i-1]*rho_l_poly[i-1]
        m_poly += [m_poly[i-1] + dm_poly_curr]
        dm_poly += [dm_poly_curr]
        # now that mass of polyol has been adjusted, estimate the volume of liquid
        V_liq += [m_poly[i]/rho_l_poly[i]]

        # initialize change in masses of components [g]
        dm_co2_curr = 0
        dm_co2_bin_curr = 0
        dm_c5_curr = 0

        # time elapsed since start of experiment [min]
        elapsed_time += [TimeDate.diff_min(timedate_start, timedate[i])]
        # identify index of current condition change by taking first index where
        # elapsed time exceeds the time of the condition change
        j = np.where(elapsed_time[i] >= elapsed_time_cc)[0][-1]
        # update condition if condition has changed
        if j == j_prev + 1:
            # compute initial and final mass in ISCO pump
            m_isco_i = rho_co2(df_cond['initial pressure (isco) [psi]'].iloc[j], T_isco, psi=True)* \
                            df_cond['initial volume (isco) [mL]'].iloc[j]
            m_isco_f = rho_co2(df_cond['final pressure (isco) [psi]'].iloc[j], T_isco, psi=True)* \
                            df_cond['final volume (isco) [mL]'].iloc[j]
            # check if ISCO contains CO2--otherwise it contains C5
            is_co2 = df_cond['co2'].iloc[j]
            is_c5 = 1 - is_co2
            # change in amount of co2 in the Parr reactor
            dm_co2_curr += is_co2*(m_isco_i - m_isco_f)
            dm_co2_bin_curr += df_cond['dm co2 [g] (bin)'].iloc[j]
            dm_c5_curr += is_c5*(m_isco_i - m_isco_f)
            # if venting, assume constant weight fractions according to previous
            # measurement and constant temperature during venting. Use CO2-C5
            # coexistence to compute vapor-phase densities
            if df_cond['venting'].iloc[j]:
                rho_v_co2_i, rho_v_c5_i = rho_v_co2_c5(df_cond['initial pressure (parr) [psi]'].iloc[j],
                                            df_cond['initial temperature (parr) [C]'].iloc[j], w_v_co2[i-1], w_v_c5[i-1])
                rho_v_co2_f, rho_v_c5_f = rho_v_co2_c5(df_cond['final pressure (parr) [psi]'].iloc[j],
                                            df_cond['final temperature (parr) [C]'].iloc[j], w_v_co2[i-1], w_v_c5[i-1])
                # adjust change in masses of vapor-phase components after venting [g]
                dm_co2_curr -= (rho_v_co2_i - rho_v_co2_f)*(V-V_liq[i])
                dm_c5_curr -= (rho_v_c5_i - rho_v_c5_f)*(V-V_liq[i])

            # update previous j
            j_prev = j

        # calculate volume of vapor sampled
        V_vap_sampled += [get_V_vap_sampled(V_vap_sampled_atm[i], w_v_co2[i], w_v_c5[i], p_parr[i], T_parr[i])]

        # estimate new masses after sampling
        dm_co2_curr -= V_vap_sampled[i-1]*rho_v_co2[i-1] + \
                    V_liq_sampled[i-1]*rho_l_co2[i-1]
        dm_c5_curr -= V_vap_sampled[i-1]*rho_v_c5[i-1] + \
                    V_liq_sampled[i-1]*rho_l_c5[i-1]
        m_co2 += [m_co2[i-1] + dm_co2_curr]
        m_co2_bin += [m_co2_bin[i-1] + dm_co2_bin_curr]
        m_c5 += [m_c5[i-1] + dm_c5_curr]
        dm_co2 += [dm_co2_curr]
        dm_co2_bin += [dm_co2_bin_curr]
        dm_c5 += [dm_c5_curr]

        # compute overall weight fractions [w/w] and densities [g/mL]
        m_tot = m_co2[i] + m_poly[i] + m_c5[i]
        w_co2 += [m_co2[i]/m_tot]
        w_poly += [m_poly[i]/m_tot]
        w_c5 += [m_c5[i]/m_tot]
        rho_tot_co2 += [m_co2[i]/V]
        rho_tot_poly += [m_poly[i]/V]
        rho_tot_c5 += [m_c5[i]/V]

        # predict mass of C5 and CO2 and compare to actual masses [g]
        m_c5_pred += [V_liq[i]*rho_l_c5[i] + (V-V_liq[i])*rho_v_c5[i]]
        m_c5_missing += [m_c5[i] - m_c5_pred[i]]
        m_co2_pred += [V_liq[i]*rho_l_co2[i] + (V-V_liq[i])*rho_v_co2[i]]
        m_co2_missing += [m_co2[i] - m_co2_pred[i]]

    # load dataframe with experimental measurements
    df['date'] = date_str
    df['time'] = time_str
    df['peak area co2 (front) [a.u.]'] = pa_f_co2
    df['peak area c5 (front) [a.u.]'] = pa_f_c5
    df['peak area co2 (back) [a.u.]'] = pa_b_co2
    df['peak area c5 (back) [a.u.]'] = pa_b_c5
    df['pressure (parr) [psi]'] = p_parr
    df['temperature (parr) [C]'] = T_parr
    df['volume sampled (vap) [mL] (atm)'] = V_vap_sampled_atm
    df['volume sampled (liq) [mL]'] = V_liq_sampled
    # load dataframe with analyzed results
    df['elapsed time [min]'] = elapsed_time
    df['mass co2 [g]'] = m_co2
    df['mass co2 [g] (bin)'] = m_co2_bin
    df['mass poly [g]'] = m_poly
    df['mass c5 [g]'] = m_c5
    df['volume liquid [mL]'] = V_liq
    df['volume sampled (vap) [mL]'] = V_vap_sampled
    df['density co2 (vap) [g/mL]'] = rho_v_co2
    df['density c5 (vap) [g/mL]'] = rho_v_c5
    df['density co2 (liq) [g/mL]'] = rho_l_co2
    df['density poly (liq) [g/mL]'] = rho_l_poly
    df['density c5 (liq) [g/mL]'] = rho_l_c5
    df['density co2 (tot) [g/mL]'] = rho_tot_co2
    df['density poly (tot) [g/mL]'] = rho_tot_poly
    df['density c5 (tot) [g/mL]'] = rho_tot_c5
    df['w co2 (vap) [w/w]'] = w_v_co2
    df['w c5 (vap) [w/w]'] = w_v_c5
    df['w co2 (liq) [w/w]'] = w_l_co2
    df['w poly (liq) [w/w]'] = w_l_poly
    df['w c5 (liq) [w/w]'] = w_l_c5
    df['w co2 (tot) [w/w]'] = w_co2
    df['w poly (tot) [w/w]'] = w_poly
    df['w c5 (tot) [w/w]'] = w_c5
    df['mass c5 (pred) [g]'] = m_c5_pred
    df['mass c5 (missing) [g]'] = m_c5_missing
    df['mass co2 (pred) [g]'] = m_co2_pred
    df['mass co2 (missing) [g]'] = m_co2_missing
    df['dm co2 [g]'] = dm_co2
    df['dm co2 [g] (bin)'] = dm_co2_bin
    df['dm poly [g]'] = dm_poly
    df['dm c5 [g]'] = dm_c5

    # add elapsed time for condition change [min]
    df_cond['elapsed time [min]'] = elapsed_time_cc

    return df


def correct_co2_leak(df, df_cond, V):
    """
    Corrects data for CO2 leak, appending new estimates to previous dataframe.

    PARAMETERS:
        df : Pandas dataframe
            Dataframe of processed results from GC sampling (see analyze_data())
        df_cond : Pandas dataframe
            Dataframe of raw data for changes in the condition (injections of
            CO2 or C5, or venting from the gas-sampling port of the Parr reactor).
        V : float
            Internal volume of Parr reactor [mL].

    RETURNS:
        df : Pandas dataframe
            Dataframe of processed results with values corrected by CO2 leak appended.
    """
    # results to be calculated
    m_co2_leak = [0]
    m_co2_corr = [0]
    m_co2_bin_corr = [0]
    rho_tot_co2_corr = [0]
    rho_tot_co2_bin_corr = [0]
    w_co2_corr = [0]
    w_co2_bin_corr = [0]
    w_c5_corr = [df['w c5 (tot) [w/w]'].iloc[0]]
    w_c5_bin_corr = [df['w c5 (tot) [w/w]'].iloc[0]]
    w_poly_corr = [df['w poly (tot) [w/w]'].iloc[0]]
    w_poly_bin_corr = [df['w poly (tot) [w/w]'].iloc[0]]
    m_co2_missing_corr = [0]
    m_co2_bin_missing_corr = [0]
    leak_rate = [0]

    # get leak rate for each condition change [g/min]
    leak = df_cond['leak co2 [g] (bin)'].to_numpy(dtype=float)
    # get elapsed time between measurements [min]
    elapsed_time = df['elapsed time [min]'].to_numpy(dtype=float)
    # get elapsed time between condition changes [min]
    elapsed_time_cc = df_cond['elapsed time [min]'].to_numpy(dtype=float)
    # compute leak rate [g/min] (pad with a zero @ beginning and end)
    leak_rate_cc = np.zeros([len(leak)+1])
    leak_rate_cc[1:-1] = leak[1:] / np.diff(elapsed_time_cc)
    # get mass of each component (uncorrected) [g]
    m_co2 = df['mass co2 [g]'].to_numpy(dtype=float)
    m_co2_bin = df['mass co2 [g] (bin)'].to_numpy(dtype=float)
    m_poly = df['mass poly [g]'].to_numpy(dtype=float)
    m_c5 = df['mass c5 [g]'].to_numpy(dtype=float)
    # get predicted mass of CO2 [g]
    m_co2_pred = df['mass co2 (pred) [g]'].to_numpy(dtype=float)

    j_prev = 0
    for i in range(1, len(df)):
        # identify index of current condition change by taking first index where
        # elapsed time exceeds the time of the condition change
        j = np.where(elapsed_time[i] >= elapsed_time_cc)[0][-1]
        leak_rate += [leak_rate_cc[j+1]]
        # estimate leak and correct CO2 mass
        m_co2_leak += [m_co2_leak[-1] + leak_rate_cc[j+1]*(elapsed_time[i]-elapsed_time_cc[j]) + \
                        leak_rate_cc[j_prev+1]*(elapsed_time_cc[j]-elapsed_time[i-1])]
        m_co2_corr += [m_co2[i] - m_co2_leak[i]]
        rho_tot_co2_corr += [m_co2_corr[i]/V]
        m_tot_corr = m_co2_corr[i] + m_poly[i] + m_c5[i]
        w_co2_corr += [m_co2_corr[i]/m_tot_corr]
        w_poly_corr += [m_poly[i]/m_tot_corr]
        w_c5_corr += [m_c5[i]/m_tot_corr]

        # use mass of co2 estimated with binary co2-c5 eos data
        m_co2_bin_corr += [m_co2_bin[i] - m_co2_leak[i]]
        rho_tot_co2_bin_corr += [m_co2_bin_corr[i]/V]
        m_tot_bin_corr = m_co2_bin_corr[i] + m_poly[i] + m_c5[i]
        w_co2_bin_corr += [m_co2_bin_corr[i]/m_tot_bin_corr]
        w_poly_bin_corr += [m_poly[i]/m_tot_bin_corr]
        w_c5_bin_corr += [m_c5[i]/m_tot_bin_corr]

        m_co2_missing_corr += [m_co2_corr[i] - m_co2_pred[i]]
        m_co2_bin_missing_corr += [m_co2_bin_corr[i] - m_co2_pred[i]]
        # update counter
        j_prev = j

    # estimate amount of cyclopentane leaked [g]
    m_c5_leak = m_co2_leak
    m_c5_leak[1:] = df['w c5 (vap) [w/w]'].to_numpy(dtype=float)[1:] / \
                    df['w co2 (vap) [w/w]'].to_numpy(dtype=float)[1:]*np.diff(m_co2_leak)
    m_c5_leak = np.cumsum(m_c5_leak)
    m_c5_corr = df['mass c5 [g]'].to_numpy(dtype=float) - m_c5_leak
    m_c5_missing_corr = df['mass c5 (missing) [g]'].to_numpy(dtype=float) - m_c5_leak

    # save results to dataframe
    df['leak rate [g/min]'] = leak_rate
    df['mass co2 (leak) [g]'] = m_co2_leak
    df['mass c5 (leak) [g]'] = m_c5_leak
    df['mass co2 [g] (corr)'] = m_co2_corr
    df['mass c5 [g] (corr)'] = m_c5_corr
    df['mass co2 [g] (bin) (corr)'] = m_co2_bin_corr
    df['mass co2 (missing) [g] (corr)'] = m_co2_missing_corr
    df['mass c5 (missing) [g] (corr)'] = m_c5_missing_corr
    df['mass co2 (missing) [g] (bin) (corr)'] = m_co2_bin_missing_corr
    df['density co2 (tot) [g/mL] (corr)'] = rho_tot_co2_corr
    df['w co2 (tot) [w/w] (corr)'] = w_co2_corr
    df['w poly (tot) [w/w] (corr)'] = w_poly_corr
    df['w c5 (tot) [w/w] (corr)'] = w_c5_corr
    df['density co2 (tot) [g/mL] (bin) (corr)'] = rho_tot_co2_bin_corr
    df['w co2 (tot) [w/w] (bin) (corr)'] = w_co2_bin_corr
    df['w poly (tot) [w/w] (bin) (corr)'] = w_poly_bin_corr
    df['w c5 (tot) [w/w] (bin) (corr)'] = w_c5_bin_corr

    return df


def correct_rho_vap(p, T, rho_v_co2_gc, rho_v_c5_gc, filename='co2_c5_vle.csv', thresh=0.4):
    """
    Corrects density measured with GC for vapor phase by using a PC-SAFT
    model fitted to experimental measurements of the CO2-C5 VLE from
    Eckert and Sandler (1986) to estimate the expected total density of the
    vapor phase of the mixture, then scales the GC measurements to have the
    same total density. Essentially, the weight fractions are kept from the
    GC measurements, but the overall density is estimated with a PC-SAFT
    model fitted to empirical data.
    """
    rho_v_gc = rho_v_co2_gc + rho_v_c5_gc
    w_v_co2 = rho_v_co2_gc / rho_v_gc
    w_v_c5 = rho_v_c5_gc / rho_v_gc
    rho_v_tot = np.sum(rho_v_co2_c5(p, T, w_v_co2, w_v_c5,
                                        filename=filename, thresh=thresh))

    # correct GC measurement
    correction = rho_v_tot/rho_v_gc
    rho_v_co2 = correction*rho_v_co2_gc
    rho_v_c5 = correction*rho_v_c5_gc

    return rho_v_co2, rho_v_c5

def get_V_vap_sampled(V_vap_sampled_atm, w_v_co2, w_v_c5, p, T):
    """
    """
    # correct mass of co2 and c5 to account for loss during gas sampling
    rho_atm = np.sum(rho_v_co2_c5(P_ATM, T_ATM, w_v_co2, w_v_c5))
    rho_parr = np.sum(rho_v_co2_c5(p, T, w_v_co2, w_v_c5))
    V_vap_sampled = (rho_atm/rho_parr) * V_vap_sampled_atm

    return V_vap_sampled


def get_calib_data(signal, compound, quantity):
    """Returns calibration data of co2 and c5 in GC."""
    # dictionary to convert code words to column names in dataframe
    col_dict = {'rho':'density [g/mL]', 'w':'weight fraction [w/w]'}
    # load data in pandas dataframe
    df = pd.read_csv('calib_' + signal + '_' + compound + '.csv')
    pa = df['peak area [a.u.]'].to_numpy(dtype=float)
    value = df[col_dict[quantity]].to_numpy(dtype=float)

    return pa, value


def pa_conv(pa, signal, compound, quantity='rho', average=True):
    """
    Loads data from calibration curves generated for the 7890 Agilent gas
    chromatograph at Dow Chemical Co., Lake Jackson, TX, and converts to a mass
    fraction or density.
    PARAMETERS:
        pa : float or numpy array
            GC peak area [a.u.] desired to be converted into a physical quantity.
        signal : string
            GC signal measured. Either 'f' (HPLIS dense-phase sampling) or
            'b' (GC light-phase sampling).
        compound : string
            Compound detected in GC. Either 'co2' (carbon dioxide) or 'c5' (cyclopentane).
        quantity : string, default='rho'
            Quantity to conver peak area to. Either 'rho' for density [g/mL] or
            'w' for weight fraction [w/w].
        average : bool, default=True
            If True, the peak areas provided will be averaged.
    RETURNS:
        result : float or numpy array
            Physical quantity requested with "quantity" parameter, same type as
            the input "pa" parameter unless averaged (in which case the
            result will be a float).
    """
    assert signal in ['f', 'b'], "Invalid signal. Choose ''f'' or ''b''."
    assert compound in ['co2', 'c5'], "Invalid compound. Choose ''co2'' or ''c5''."
    x, y = get_calib_data(signal, compound, quantity)
    # perform linear fit to get calibration conversion
    a, b = np.polyfit(x, y, 1)
    # estimate quantity for given peak area
    result = a*pa + b
    # average results
    if average:
        result = np.mean(result)

    return result


def rho_co2(p, T, eos_file_hdr='eos_co2_', ext='.csv', psi=False):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at 30.5 C.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.
    PARAMETERS:
        p : int (or array of ints)
            pressure in kPa of CO2 (unless psi==True)
        T : float
            temperature in Celsius (only to one decimal place)
        eos_file_hdr : string, default='eos_co2_'
            File header for equation of state data table
        ext : string, default='.csv'
            Extension for file, including period (4 characters).
        psi : bool, default=False
            If True, treats input pressure as psi.
    RETURNS:
        rho : same as p
            density in g/mL of co2 @ 30.5 C
    """
    # convert to kPa if pressure is passed as psi
    if psi:
        p *= 100/14.5
    # get decimal and integer parts of the temperature
    dec, integ = np.modf(T)
    # create identifier string for temperature
    T_tag = '%d-%dC' % (integ, 10*dec)
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(eos_file_hdr + T_tag + ext, header=0)
    # get list of pressures of all data points [kPa]
    p_co2_kpa = df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [g/mL]
    rho_co2 = df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # remove repeated entries
    p_co2_kpa, inds_uniq = np.unique(p_co2_kpa, return_index=True)
    rho_co2 = rho_co2[inds_uniq]
    # create interpolation function and interpolate density [g/mL]
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")
    rho = f_rho(p)

    return rho


def rho_v2110b(T):
    """
    Estimates density of VORANOL 2110B polyol at given temperature based on
    data collected at Dow for P-1000 (also 1000 g/mol, difunctional polyol).
    PARAMETERS:
        T : float
            Temperature [C].
    RETURNS:
        rho : float
            Density [g/mL]
    """
    T_data = np.array([25, 50, 75, 100])
    rho_data = np.array([1.0015, 0.9826, 0.9631, 0.9438])
    # estimate density with linear regression
    a, b = np.polyfit(T_data, rho_data, 1)
    rho = a*T + b

    return rho


def rho_liq(rho_c, T_c, A, B, C, D, T):
    """
    Empirical equation for the density of a liquid from equation 1 of Chapter D3,
    "Properties of Pure Fluid Substances," by Michael Kleiber and Ralph Joh,
    Section 1, "Liquids and Gases," of the VDI Heat Atlas (2010).
    PARAMETERS:
        rho_c : float
            Critical density [kg/m^3].
        T_c : float
            Critical temperature [K].
        A, B, C, D : floats
            fitting parameters of equation
        T : float or (N x 1) numpy array
            Desired temperature(s) [K].
    RETURNS:
        rho_liq : float or (N x 1) numpy array (same as T)
            Density of desired temperature(s) [kg/m^3].
    """
    rho_liq = rho_c + A*(1-T/T_c)**(0.35) + B*(1-T/T_c)**(2/3) + C*(1-T/T_c) + \
                D*(1-T/T_c)**(4/3)
    return rho_liq


def rho_c5(T, empirical_formula=False):
    """
    Estimates density of cyclopentane below critical temperature with empirical
    equation for liquid-phase densities in Chapter D3, "Properties of Pure Fluid
    Substances," by Michael Kleiber and Ralph Joh, Section 1, "Liquids and Gases,"
    of the VDI Heat Atlas (2010). The equation is of the form:

    rho_liq = rho_c + A*(1-T/T_c)**(0.35) + B*(1-T/T_c)**(2/3) + C*(1-T/T_c) +
                D*(1-T/T_c)**(4/3)

    SOURCE:
    Michael Kleiber, Ralph Joh, Roland Span (2010)
    SpringerMaterials
    D3 Properties of Pure Fluid Substances
    VDI-Buch
    (VDI Heat Atlas)
    https://materials-springer-com.clsproxy.library.caltech.edu/lb/docs/sm_nlb_978-3-540-77877-6_18
    10.1007/978-3-540-77877-6_18 (Springer-Verlag © 2010)
    Accessed: 04-10-2019

    PARAMETERS:
        T : float or (N x 1) numpy array
            Temperature(s) [C].
    RETURNS:
        rho_c5 : float or (N x 1) numpy array (same as T)
            Density of cyclopentane at desired temperature(s) [g/mL].
    """
    if empirical_formula:
        A, B, C, D = [450.0150, 692.7447, -1131.7809, 724.4630]
        rho_c = 270 # [kg/m^3]
        T_c = 511.7 # [K]
        rho_c5 = rho_liq(rho_c, T_c, A, B, C, D, T+273.15)/1000 # [g/mL]
    else:
        T_data = np.array([0, 20, 50, 100]) # [C]
        rho_data = np.array([0.7652, 0.7465, 0.7176, 0.6652]) #[g/mL]
        # a, b = np.polyfit(T_data, rho_data, 1)
        # rho_c5 = a*T + b # [g/mL]
        rho_c5_f = interp1d(T_data, rho_data)
        rho_c5 = rho_c5_f(T)

    return rho_c5


def rho_v_co2_c5(p, T, w_v_co2, w_v_c5, filename='co2_c5_vle.csv', thresh=0.4):
    """
    Estimates density of co2-c5 binary vapor mixture by using a PC-SAFT
    model fitted to experimental measurements of the CO2-C5 VLE from
    Eckert and Sandler (1986) to estimate the expected total density of the
    vapor phase of the mixture.
    Pressure in psi, temperature in C.
    """
    # load dataframe of co2-c5 VLE
    df = pd.read_csv(filename)
    # select temperature
    T_arr = df['temperature [C]'].to_numpy(dtype=float)
    # remove duplicate temperatures
    T_uniq = np.unique(T_arr)
    # choose the temperature in the array that is closest to the current temperature
    T_nearest = T_uniq[int(np.argmin(np.abs(T_uniq-T)))]
    # extract entries for selected temperature
    inds = np.where(T_arr==T_nearest)[0]
    p_arr = df['pressure [psi]'].to_numpy(dtype=float)[inds]
    w_v_co2_arr = df['w co2 vap [w/w]'].to_numpy(dtype=float)[inds]
    w_v_c5_arr = df['w c5 vap [w/w]'].to_numpy(dtype=float)[inds]
    rho_v_co2_arr = df['density co2 vap [g/mL]'].to_numpy(dtype=float)[inds]
    rho_v_c5_arr = df['density c5 vap [g/mL]'].to_numpy(dtype=float)[inds]
    # interpolate weight fractions and densities
    w_v_co2 = np.interp(p, p_arr, w_v_co2_arr)
    w_v_c5 = np.interp(p, p_arr, w_v_c5_arr)
    rho_v_co2 = np.interp(p, p_arr, rho_v_co2_arr)
    rho_v_c5 = np.interp(p, p_arr, rho_v_c5_arr)
    # ensure that the weight fractions match what the GC measured
    assert np.max(np.abs(w_v_co2-w_v_co2)) < thresh and \
            np.max(np.abs(w_v_c5-w_v_c5)) < thresh, "weight fractions too different." + \
            "{0:.3f} vs. {1:.3f} for {2:d} psi and {3:d} C.".format(w_v_co2, w_v_co2, p, T)

    return rho_v_co2, rho_v_c5


def rho_polyol_co2(p, T):
    """Estimates density of polyol-CO2 mixture based on Naples data."""
    T_list = [30.5, 60]
    if T < min(T_list):
        return 1.02
    df_30 = pd.read_csv('1k2f_30c.csv')
    df_60 = pd.read_csv('1k2f_60c.csv')
    p_arr_list = [df_30['p actual [kPa]'].to_numpy(dtype=float),
                    df_60['p actual [kPa]'].to_numpy(dtype=float)]
    spec_vol_arr_list = [df_30['specific volume [mL/g]'].to_numpy(dtype=float),
                    df_60['specific volume [mL/g]'].to_numpy(dtype=float)]
    spec_vol_list = [np.interp(p, np.array(p_arr_list[i]),
                    np.array(spec_vol_arr_list[i])) for i in range(len(p_arr_list))]
    spec_vol = np.interp(T, np.array(T_list), np.array(spec_vol_list))
    rho = 1/spec_vol

    return rho
