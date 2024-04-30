#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Routines related to the atmospheric effects for use in astronomy

Copyright (C) 2024 Wing-Fai Thi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Wing-Fai Thi, wingfai.thi@googlemail.com

History:
    2024/04/30 version 1.0
"""
# third-party packages numpy and astropy
import numpy as np
import astropy.units as u


def beta_stone(temperature):
    """
    Return the beta factor in the 2 term expansion
    of the atmosphere refraction model eq. 9

    beta = H / R, where H is the Earth atmosphere scale height
    and R is the Earth radius. The default value is
    H = 7.99 km and R = 6370.9 km.

    The local scale-height depends on the temperature

    References
    ----------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    Parameter
    ---------
    temp : `astropy.units.quantity.Quantity` degree Celsius
        atmospheric temperature in degrees C
        The standard temperature is 10 °C

    returns
    -------
    beta : float
        the value of beta

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> temp = 10 * u.deg_C
    >>> atm.beta_stone(temp)
    0.0012999088412959912
    """
    t = temperature.to(u.deg_C).value
    return 0.001254 * (273.15 + t) / 273.15


def kappa_stone(latitude, elevation):
    """
    Return the kappa factor in the 2 term expansion
    of the atmosphere refraction model eq. 10

    κ (kappa) is defined as the ratio of the
    gravity go at the observing site to the sea-level
    gravity g at the Earth's equator.

    References
    ----------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
        Refraction," Publications of the Astronomical Society of the
        Pacific, vol. 108, p. 1051, 1996.

    Parameters
    ----------
    latitude : `astropy.units.quantity.Quantity` degree
        the latitude of the observer

    elevation : `astropy.units.quantity.Quantity` m
        the elevation of the observer

    Returns
    -------
    kappa : float
        the value of kappa

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> lat = -24.615833 * u.deg
    >>> elv = 2518. * u.m
    >>> atm.kappa_stone(lat, elv)
    1.0004967193501944
    """
    lat = latitude.to(u.deg).value
    h = elevation.to(u.meter).value
    kappa = 1 + 0.005302 * np.sin(lat)**2
    kappa -= 0.00000583 * np.sin(2 * lat)
    kappa -= 0.000000315 * h
    return kappa


def standard_pressure(height):
    """
    Compute the standard air pressure given the height in meters

    Parameter
    ---------
    height : `astropy.units.quantity.Quantity` length (meters)
        the height of the observer (telescope location)
        This is sometime called elevation

    Returns
    -------
    pressure : `astropy.units.quantity.Quantity` pressure (Pa)
        the standard air pressure at height m

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> atm.standard_pressure(2518 * u.m)
    <Quantity 75183.76915713 Pa>

    Notes
    -----
    tsl is the approximate sea-level air temperature in degrees K
    see Astrophysical Quantities, C.W.Allen)
    1 millibar = 100 Pa

    The equation is taken from SLALIB
    """
    tsl = 288.0  # Kelvin and pressure = 101,325 Pa
    height = height.to(u.meter)
    pressure = 1013.25 * np.exp(-height.value / (29.3 * tsl))  # millibar
    pressure *= 100
    pressure *= u.Pa
    return pressure


def refrac_std_iau(alt, height=None, wave=None):
    """
    Compute the observed altitude at different wavelengths
    with a standard atmosphere as adopted by IAU

    Parameters
    ----------
    alt : `astropy.units.quantity.Quantity` degrees
        the altitude without the refraction (air dispersion correction
        applied to it

    height : `astropy.units.quantity.Quantity` length (meters), optional,
             default = None
        the height of the observer (telescope location), aka elevation

    wave : `astropy.units.quantity.Quantity` (recommended) micron, optional
        single value of array of wavelengths
        if no value is entered, the wavelength is set by default to 0.56 micron

    Returns
    -------
    `astropy.units.quantity.Quantity` degrees
        the observed alttude

     `astropy.units.quantity.Quantity`
        the air refraction R

    Notes
    -----
    It calls almanac.airindex_Owen(wave, pressure, temp, relhum)

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> atm.refrac_std_iau(65 * u.deg, wave=0.56 * u.micron)
    (<Quantity 65.00751267 deg>, <Quantity 0.00751267 deg>)
    >>> height = 2500 * u.meter
    >>> atm.refrac_std_iau(65 * u.deg, height=height,
    ...                    wave=0.56 * u.micron)
    (<Quantity 65.00560326 deg>, <Quantity 0.00560326 deg>)
    """
    if height is None:
        pressure = 101.0 * 1e3 * u.Pa
    else:
        pressure = standard_pressure(height)
    temp = 10 * u.deg_C
    relhum = 10  # relative humidity in percent
    if wave is None:
        wave = 0.55 * u.micron
    n = airindex_Owen(wave, pressure, temp, relhum)
    zref, refrac = refrac_iau(90 * u.deg - alt, n)
    return 90 * u.deg - zref, refrac


def refrac_iau(z, n, beta=7.99 / 6370.9, kappa=1, five=False):
    """
    Apply atmospheric refraction to an apparent 'true'
    zenith angle to obtain the observed zenith angle

    The routine uses the classic refraction formula with two tan(z)
    terms. This does not assume any information about the structure
    of the atmosphere.

    It uses a two term expansion of the refraction integral [1]

    The formula is valid for zenight angle < 75 degrees or 25 degrees
    altitude/elevation

    A genereal discussion is provided by [3]

    References
    ----------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    [2] Corbard, T. et al. MNRAS 483, 3865–3877, 2019

    [3] Kovalevsky Modern Astrometry, Springer 2002 Chap. 3

    Parameters
    ----------
    n : float or numpy array of float
        the index of refraction for the input conditions.

    z : `astropy.units.quantity.Quantity` degree
        true zenith angle (as if there were no atmosphere)

    five : bool, optional, default=False
        if true, use a power to the 5 expansion, see eq. 6 in [2]

    beta : float, optional, default = h / R
        the value of beta eq. 9 in [1]
        h = 7.99 * u.km  # height of the homogeneous atmosphere
        R = 6370.9 * u.km  # mean radius of the Earth

    kappa: float, optional, default =1 (spherical Earth)
        the value of kappa eq. 10 in [1]

    Returns
    -------
    zref : `astropy.units.quantity.Quantity` degree
        the observed zenith angle

    refrac : astropy.units.quantity.Quantity` degree
        the refraction

    Example
    -------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.37, 0.56, 0.95] * u.micron
    >>> pressure = 101.0 * 1e3 * u.Pa
    >>> temp = 10 * u.deg_C
    >>> relhum = 10  # percent
    >>> n = atm.airindex_Filippenko(wave, pressure, temp, relhum)
    >>> zref, refrac = atm.refrac_iau(30 * u.deg, n)
    >>> zref
    <Quantity [29.99046712, 29.99070022, 29.99081145] deg>
    >>> dzref = (zref-zref[1]).to(u.arcsec)
    >>> dzref
    <Quantity [-0.83916744,  0.        ,  0.40043794] arcsec>
    >>> beta = atm.beta_stone(10 * u.deg_C)
    >>> kappa = atm.kappa_stone(-24.615833 * u.deg, 2518. * u.m)
    >>> zref, refrac = atm.refrac_iau(30 * u.deg, n, beta=beta, kappa=kappa)
    >>> wave = np.arange(0.37, 0.9, 0.01) * u.micron
    >>> n = atm.airindex_Filippenko(wave, pressure, temp, relhum)
    >>> zref, refrac = atm.refrac_iau(60 * u.deg, n, beta=beta, kappa=kappa)
    >>> refrac[0]
    <Quantity 0.02852279 deg>
    """
    n1 = n - 1
    refrac = kappa * n1 * (1 - beta) * np.tan(z)
    refrac -= kappa * n1 * (beta - 0.5 * n1) * np.tan(z)**3
    if (five):
        refrac += 3 * kappa * n1 * (beta - 0.5 * n1)**2 * np.tan(z)**5
    refrac *= u.rad
    refrac = refrac.to(u.deg)
    return z - refrac, refrac


def refrac_Buie(z, n):
    """
    Apply atmospheric refraction to an apparent 'true'
    zenith angle to obtain the observed zenith angle

    Based on Marc Buie refrac.pro

    Paramater
    ---------
    n : float or numpy array of float
        return value is the index of refraction for the input conditions.

    z : `astropy.units.quantity.Quantity` degree
        true (aka apparent) zenith angle (as if there were no atmosphere)

    Returns
    -------
    zref : `astropy.units.quantity.Quantity` degree
        the observed zenith angle

    refrac : astropy.units.quantity.Quantity` degree
        the refraction

    Notes
    -----
    From Marc Buie:
    This calculation is based on a few different sources.  First, it is
    assumed that the index of refraction of air at the base of the atmosphere
    can be calculated (see AIRINDEX).  From the index of refraction, the
    bending is computed from the formula on p.55 of the old Explanatory
    Supplment to the Nautical Almanac.  This formula has been modified by
    removing the h/rho term.
    The explanatory supplement doesn't indicate that this is
    legitimate but I've validated this computation against a more emperical
    formalism from Eisele and Shannon (NRL memo 3058, May 1975).  Eisele and
    Shannon don't indicate the wavelength of light used but if I use
    0.56 microns and compare for the same input conditions (dry air only),
    the refraction computed agrees to within 1 arcsec down to 51 degrees zenith
    angle and is good to 10 arcsec down to 80 degrees.

    Restrictions
    ------------
    Not accurate (nor useful) for z > 85 degrees.

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.37, 0.56, 0.95] * u.micron  # 370 – 950 nm
    >>> pressure = 101.0 * 1e3 * u.Pa
    >>> temp = 10 * u.deg_C
    >>> relhum = 10  # percent
    >>> n = atm.airindex_Filippenko(wave, pressure, temp, relhum)
    >>> zref, refrac = atm.refrac_Buie(30 * u.deg, n)
    >>> zref
    <Quantity [29.99045482, 29.99068814, 29.99079947] deg>
    >>> dzref = (zref-zref[1]).to(u.arcsec)
    >>> dzref
    <Quantity [-0.83993416,  0.        ,  0.40080937] arcsec>
    """
    zref = np.arcsin(np.sin(z) / n).to(u.deg)
    refrac = z - zref
    return zref, refrac


def airindex_Filippenko(wave, pressure, temp, relhum):
    """
    Compute the real part of the refractive index of air.

    This function is based on the formulas in Filippenko, 1982 PASP, v. 94,
    pp. 715-721 for the index of refraction of air.  The conversion from
    relative humidity to vapor pressure is from the Handbook of Chemistry
    and Physics.

    based on airindex.pro by Marc W. Buie, STScI, 2/28/91

    The refraction is based on a flat atmosphere and the refractivity
    values are calculated from a modified Edlen (1953) equation.

    Parameter
    ---------
    wave : `astropy.units.quantity.Quantity` microns
        wavelength of light, in microns

    pressure : `astropy.units.quantity.Quantity` Pascal
        atmospheric pressure in mm of Hg
        1 Pa = 0.00750062 mmHg
        The standard value is 101.0 kPa

    temp : `astropy.units.quantity.Quantity` degree Celsius
        atmospheric temperature in degrees C
        The standard temperature is 10 °C

    relhum : float
        Relative humidity (in percent)

    Returns
    -------
    n : float or numpy array of float
        return value is the index of refraction for the input conditions.

    References
    ----------
    [1] Filippenko, 1982 PASP, v. 94, pp. 715-721

    [2] B. Edlén "The refractive index of air", Metrologia  2, 71–80 (1966)

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.37, 0.56, 0.95] * u.micron  # 370 – 950 nm
    >>> pressure = 101.0 * 1e3 * u.Pa
    >>> temp = 10 * u.deg_C
    >>> relhum = 10  # percent
    >>> n = atm.airindex_Filippenko(wave, pressure, temp, relhum)
    >>> n
    array([1.00028865, 1.00028159, 1.00027822])
    """
    wave = np.array(wave.to(u.micron).value)
    pressure = pressure.to(u.Pa).value * 0.00750062  # conversion to mm Hg
    temp = temp.to(u.deg_C).value

    # Eden's air index of refraction at wavelength wave
    n = 64.328 + 29498.1 / (146.0 - (1.0 / wave)**2) + 255.4\
        / (41.0 - (1.0 / wave)**2)
    pfac = pressure * (1.0 + (1.049 - 0.0157 * temp) * 1.0e-6 * pressure)\
        / (720.883 * (1.0 + 0.003661 * temp))
    dt = 100.0 - temp
    logp = 2.8808 - 5.67 * dt / (274.1 + temp - 0.15 * dt)
    f = (relhum / 100) * 10**logp
    water = (0.0624 - 0.000680 / wave**2) * f / (1.0 + 0.003661 * temp)
    n = (n - water) * pfac
    n = 1.0 + n * 1.0e-6
    return n


def vapor_pressure(temp, relhum):
    """
    Compute the vapor pressure given the temperature (in C) and
    relative humidity

    see 18 & 20 of Ref. 1

    Reference
    ---------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    Parameter
    ---------
    temp : `astropy.units.quantity.Quantity` degree Celsius
        atmospheric temperature in degrees C
        The standard temperature is 10 °C
        The value should be -23° C < temp < 47° C

    relhum : float
        Relative humidity (in percent) 0 < relhum < 100%

    Returns
    -------
    pw_mm : float
        water vapor pressure in units of mm Hg

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> temp = 10 * u.deg_C
    >>> relhum = 10  # 10%
    >>> atm.vapor_pressure(temp, relhum)
    0.7607021256749102
    """
    x = np.log(relhum / 100)
    a = 238.3
    b = 17.2694
    t = temp.to(u.deg_C).value
    td_C = (t + a) * x + b * t
    td_C /= (t + a) * (b - x) - b * t
    td_C *= a

    coeff = np.array([0.203447e-7, 0.238294e-5, 0.184889e-3,
                      0.0106778, 0.341724, 4.50874])
    pw_mm = np.polyval(coeff, td_C)

    if (relhum < 1e-4):
        pw_mm = 0.0

    return pw_mm


def airindex_Owen(wave, pressure, temp, relhum):
    """
    Compute the air refraction index according to Owen's model
    (1967) [1, 2]

    see ref 1 eq. 14 - 17

    Parameters
    ----------
    wave : `astropy.units.quantity.Quantity` microns
        wavelength of light, in microns
        2302 Â < wave <20,586 Â

    pressure : `astropy.units.quantity.Quantity` Pascal
        atmospheric pressure in mm of Hg
        1 Pa = 0.00750062 mmHg
        The standard value is 101.0 kPa
        Valid values are in the range 0 < ps < 4 atm

    temp : `astropy.units.quantity.Quantity` degree Celsius
        atmospheric temperature in degrees C
        The standard temperature is 10 °C
        The value should be -23° C < temp < 47° C

    relhum : float
        Relative humidity (in percent) 0 < relhum < 100%

    Returns
    -------
    n : float or numpy array of float
        return value is the index of refraction for the input conditions.

    Reference
    ----------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    [2] J. C. Owens "Optical refractive index of air: dependence on pressure,
         temperature and composition", Appl. Opt.  6, 51–59 (1967)

    Notes
    -----
    This air index model is used by SOFA IAU)

    Example
    -------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.37, 0.56, 0.95] * u.micron
    >>> pressure = 101e3 * u.Pa
    >>> temp = 10 * u.deg_C
    >>> relhum = 10
    >>> n1 = atm.airindex_Owen(wave, pressure, temp, relhum)
    >>> n2 = atm.airindex_Filippenko(wave, pressure, temp, relhum)
    >>> n1
    array([1.00028866, 1.00028161, 1.00027825])
    >>> n2
    array([1.00028865, 1.00028159, 1.00027822])
    >>> wave = [370, 480, 620, 860, 960, 1025]  * u.nm
    >>> lsstLat = -30.244639 * u.deg
    >>> lsstLon = -70.749417 * u.deg
    >>> lsstAlt = 2663. * u.m
    >>> temp = 10 * u.deg_C  # in degrees Celsius
    >>> relhum = 10  # in percent
    >>> pressure = 73892. * u.Pa  # 1 atmosphere.
    >>> alt = ((np.pi / 6.) * u.rad).to(u.deg)
    >>> n1 = atm.airindex_Owen(wave, pressure, temp, relhum)
    >>> beta = atm.beta_stone(10 * u.deg_C)
    >>> kappa = atm.kappa_stone(lsstLat, lsstAlt)
    >>> zref, refrac = atm.refrac_iau(alt, n1, beta=beta, kappa=kappa)
    >>> zref[0:2]
    <Quantity [29.99300173, 29.99312624] deg>

    For reference
    https://emtoolbox.nist.gov/Wavelength/Ciddor.asp
    n = 1.000281613 at 0.56n micron with CO2 content of 450 Micromole per
        Mole [parts per million, ppm]

    n = 1.00028161 using the elden formula
    """
    ps = pressure.to(u.Pa).value * 0.00750062  # conversion to mm Hg
    t = 273.15 + temp.to(u.deg_C).value  # T in Kelvin
    pw = vapor_pressure(temp, relhum)  # in mm Hg

    Ps = 1.333224 * (ps - pw)  # Ps is in millibars
    Pw = 1.333224 * pw  # Pw is in millibars

    tt = 1 / t

    Ds = 1 + Ps * (57.9e-8 - 9.325e-4 * tt + 0.25844 * tt**2)
    Ds *= Ps * tt

    Dw = Pw * (1 + 3.7e-4 * Pw)
    Dw *= (-2.37321e-3 + 2.23366 * tt - 710.92 * tt**2 + 7.75141e4 * tt**3)
    Dw += 1
    Dw *= Pw * tt

    sig = 1 / wave.to(u.micron).value  # micron^(-1)
    sig2 = sig * sig

    n1 = (2371.34 + 683939.7 / (130 - sig2) + 4547.2 / (38.9 - sig2)) * Ds
    n1 += (6487.31 + 58.058 * sig2 - 0.71150 * sig2**2 + 0.08851 * sig2**3) *\
        Dw
    n = 1e-8 * n1 + 1.0

    return n


def airindex_Peck_Reeder(wave):
    """
    Refractive index of the air [1]. This simple formula corrects some errors
    in the IR in the Eden formula

    Standard air: dry air at 15 °C, 101.325 kPa and with 450 ppm CO2 content.

    Parameters
    ----------
    wave : `astropy.units.quantity.Quantity` microns
        wavelength of light, in microns

    Returns
    -------
    n : float or numpy array of float
        return value is the index of refraction at wavelength wave

    Reference
    ---------
    [1] E. R. Peck and K. Reeder. Dispersion of air, J. Opt. Soc. Am. 62,
    958-962 (1972)

    [2] https://refractiveindex.info/?shelf=other&book=air&page=Peck

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.35, 0.55, 0.95] * u.micron
    >>> n = atm.airindex_Peck_Reeder(wave)
    >>> atm.airindex_Peck_Reeder(0.5876 * u.micron)
    1.0002771595071915
    >>> # 0.5876 micron -> 1.00027716 from the web siite [2]
    """
    wave = wave.to(u.micron).value
    inv_wave2 = 1 / wave**2
    n1 = 8.06051e-5
    n1 += 2.480990e-2 / (132.274 - inv_wave2)
    n1 += 1.74557e-4 / (39.32957 - inv_wave2)
    return n1 + 1


def drefrac_dz(z0, z1, n, beta=7.99 / 6370.9, kappa=1):
    """
    Differential refraction between two zenith angles
    z0 and z1 [1]. [1] claims an accuracy down to 0".001
    up to a zenith distance (angle) z0 of 70 degrees small
    small difference in altitude

    Parameters
    ----------
    n : float or numpy array of float
        the index of refraction for the input conditions.

    z0 : `astropy.units.quantity.Quantity` degree
        the first true zenith angle (as if there were no atmosphere)

    z1 : `astropy.units.quantity.Quantity` degree
        the second zenith angle. z1 - z0 is a small value

    beta : float, optional, default = h / R
        the value of beta eq. 9 in [2]
        h = 7.99 * u.km  # height of the homogeneous atmosphere
        R = 6370.9 * u.km  # mean radius of the Earth

    kappa: float, optional, default =1 (spherical Earth)
        the value of kappa eq. 10 in [2]

    Returns
    -------
        dR : `astropy.units.quantity.Quantity` degree
            the differential refraction

    Reference
    ---------
    [1] Kovalevsky Modern Astrometry, Springer 2002 Chap. 3.1.6

    [2] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    Examples
    --------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = 0.55 * u.micron
    >>> n = atm.airindex_Peck_Reeder(wave)
    >>> dz = 1 * u.arcmin
    >>> z0 = 50 * u.deg
    >>> z1 = z0 + dz
    >>> dr = atm.drefrac_dz(z0, z1, n)
    >>> zref0, refrac0 = atm.refrac_iau(z0, n)
    >>> zref1, refrac1 = atm.refrac_iau(z1, n)
    >>> refrac0 - refrac1
    <Quantity -1.11433294e-05 deg>
    """
    dz = ((z0 - z1).to(u.rad)).value
    n1 = n - 1
    A = kappa * n1 * (1 - beta)
    B = kappa * n1 * (beta - 0.5 * n1)
    tanz0 = np.tan(z0)
    dr = (A - 3 * B * tanz0**2) * dz
    dr += (A * tanz0 - B * (3 * tanz0 + 6 * tanz0**3)) * dz**2
    dr += (1. / 3.) * (A * (1 + 3 * tanz0**2) -
                       B * (3 + 27 * tanz0**2 + 30 * tanz0**4)) * dz**3
    dr *= (1 + tanz0**2)
    dr *= u.rad

    return dr.to(u.deg)


def drefrac_iau(z0, z1, n, beta=7.99 / 6370.9, kappa=1):
    """
    Differential refraction between two zenith angles
    z0 and z1 [1]

    Parameters
    ----------
    n : float or numpy array of float
        the index of refraction for the input conditions.

    z0 : `astropy.units.quantity.Quantity` degree
        the first true zenith angle (as if there were no atmosphere)

    z1 : `astropy.units.quantity.Quantity` degree
        the second zenith angle.

    beta : float, optional, default = h / R
        the value of beta eq. 9 in [2]
        h = 7.99 * u.km  # height of the homogeneous atmosphere
        R = 6370.9 * u.km  # mean radius of the Earth

    kappa: float, optional, default =1 (spherical Earth)
        the value of kappa eq. 10 in [2]

    Returns
    -------
        dR : `astropy.units.quantity.Quantity` degree
            the differential refraction

    Reference
    ---------
    [1] R. C. Stone, "An Accurate Method for Computing Atmospheric
       Refraction," Publications of the Astronomical Society of the Pacific,
       vol. 108, p. 1051, 1996.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = 0.55 * u.micron
    >>> n = atm.airindex_Peck_Reeder(wave)
    >>> # dz_list = [0.1, 1, 2, 5, 10] * u.arcmin
    >>> dz_list = [0.1, 0.5, 1, 1.5, 2, 2.5] * u.deg
    >>> z0 = np.arange(0,70.1,1) * u.deg
    >>> dr = []

    # Extra example
    for dz in dz_list:
        z1 = z0 + dz
        _, dr = atm.drefrac(z0, z1, n)
        plt.plot(z0, np.abs(dr.to(u.arcsec)),label='sep =' + str(dz))
    plt.title('Differential atmospheric refraction')
    plt.xlabel('zenith angle z (degrees)')
    plt.ylabel('dz (arcsec)')
    plt.legend()
    plt.show()  #doctest: +SKIP
    plt.close()
    """
    zref0, refrac0 = refrac_iau(z0, n)
    zref1, refrac1 = refrac_iau(z1, n)
    dz = zref0 - zref1
    dr = refrac0 - refrac1
    return dz, dr


def refracStone(z, wave, pressure, temp, relhum, latitude, elevation):
    """
    Complete Stone refraction model

    Parameters
    ----------
    z : `astropy.units.quantity.Quantity` degree
        true zenith angle (as if there were no atmosphere)

    wave : `astropy.units.quantity.Quantity` microns
        wavelength of light, in microns
        2302 Â < wave <20,586 Â

    pressure : `astropy.units.quantity.Quantity` Pascal
        atmospheric pressure in mm of Hg
        1 Pa = 0.00750062 mmHg
        The standard value is 101.0 kPa
        Valid values are in the range 0 < ps < 4 atm

    temp : `astropy.units.quantity.Quantity` degree Celsius
        atmospheric temperature in degrees C
        The standard temperature is 10 °C
        The value should be -23° C < temp < 47° C

    relhum : float
        Relative humidity (in percent) 0 < relhum < 100%

    latitude : `astropy.units.quantity.Quantity` degree
        the latitude of the observer

    elevation : `astropy.units.quantity.Quantity` m
        the elevation of the observer

    Returns
    -------
    zref : `astropy.units.quantity.Quantity` degree
        the observed zenith angle

    refrac : astropy.units.quantity.Quantity` degree
        the refraction

    Example
    -------
    >>> import astropy.units as u
    >>> import atmosphere as atm
    >>> wave = [0.37, 0.56, 0.95] * u.micron
    >>> pressure = 101.0 * 1e3 * u.Pa
    >>> temp = 10 * u.deg_C
    >>> relhum = 10  # percent
    >>> latitude = -24.615833 * u.deg
    >>> elevation = 2518. * u.m
    >>> z = 60 * u.deg
    >>> zref, refrac = atm.refracStone(z, wave, pressure,
    ...                                temp, relhum, latitude, elevation)
    >>> zref
    <Quantity [59.97147587, 59.97217284, 59.97250482] deg>
    >>> z = [45, 60, 80] * u.deg
    >>> zref, refrac = atm.refracStone(z, wave, pressure,
    ...                                temp, relhum, latitude, elevation)
    """
    n = airindex_Owen(wave, pressure, temp, relhum)
    beta = beta_stone(10 * u.deg_C)
    kappa = kappa_stone(latitude, elevation)
    if z.isscalar:
        zref, refrac = refrac_iau(z, n, beta=beta, kappa=kappa)
    else:
        zref, refrac = [], []
        for zval in z:
            zr, ref = refrac_iau(zval, n, beta=beta, kappa=kappa)
            zref.append(zr.value)
            refrac.append(ref.value)
        zref *= u.deg
        refrac *= u.deg
    return zref, refrac


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
