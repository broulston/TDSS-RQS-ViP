# TDSS Repeat Quasar Spectroscopy (RQS) Vi

[![GitHub release](https://img.shields.io/github/release-pre/broulston/TDSS-RQS-ViP.svg)](https://github.com/broulston/TDSS-RQS-ViP/releases/latest)
[![GitHub commits](https://img.shields.io/github/commits-since/broulston/TDSS-RQS-ViP/v0.0.svg)](https://github.com/broulston/TDSS-RQS-ViP/commits/master)
[![GitHub issues](https://img.shields.io/github/issues/broulston/TDSS-RQS-ViP.svg)](https://github.com/broulston/TDSS-RQS-ViP/issues)
[![license](https://img.shields.io/github/license/broulston/TDSS-RQS-ViP.svg)](https://github.com/broulston/TDSS-RQS-ViP/blob/master/license.txt)
[![Python Supported](https://img.shields.io/badge/Python%20Supported-3-brightgreen.svg)](conda)
[![Maintenance](https://img.shields.io/maintenance/yes/2019.svg)]()
[![Powered by Astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)

`TDSS Vi` creates one (1) plot for each of the variable stars in the program sample. These plots are designed  to help summarize  as much information about each star as possible, and to aid in classification. Each plot currently contains: 

![Vi_plot](./000.04023+005.77216_Vi.png?raw=true)

1. CSS Light curve (in raw form if `logProb_Per > -10.0`, and in folded phase if `logProb_Per <= -10.0`)
2. Some general population plot based on if the star is periodic or not:
    * Plot from Palaversa et al. (2013) if periodic. This is `log10(P/days)` on the x-axis and `log10(Amp/mag)` on the y-axis with light curve skewness as the colorbar.
    * Generic _heat map_ of `a95` vs log10($\chi^2$) if star is non-periodic.
3. Color-Magnitude diagram based on SDSS colors and Gaia DR2 distances.
4. SDSS spectrum with properties printed (this includes information from PyHammer)

These plots are created using the `Vi` code written by Ben Roulston (BU/SAO). In addition to these plots, the Vi code also creates a `.csv` file that contains information about the light curves for each star from the `Vartools` program. The `Vartools` program also saves the phase folded light curve for **EVERY** star as well (even though it is not plotted or used for every star). 

The main functions of the program (which call `Vartools` and creates the figures) are in the `VarStar_Vi_plot_functions.py` file. This notebook shows how this Vi program should be run. In order to run this Vi program, the following items are needed and **MUST** be downloaded/created/known before any figures can be made.

1. CSS light curves for every star.
2. File connecting the `CSS_ID` to the `RA` and `DEC` for each star. (A simple `.csv` that is just `[ra, dec, css_id] works fine and is what is used here.
3. SDSS spectrum for every star.
    * If the star has been in a public data release, then the SDSS `.fits` format is fine (i.e. `spec-MJD-PLATE-FIBERID.fits`)
    * If the star has **NOT** been in a public data release (i.e. is proprietary) then you need to download the `spPlate-PLATE-MJD.fits` for each. Then pull out the spectra and save them in ASCII format.
4. Know where all data is stored and make sure the directories in the code are up to date (this is **VERY** important).
5. Property table which links **ALL** spALL data with Gaia DR2 data and CSS data.
6. Installation of the `Vartools` program for analyzing light curves.

Once all of this is ready, the `Vi` program can be run. 

`Vi` uses the [Vartools](http://adsabs.harvard.edu/abs/2016arXiv160506811H) program, which was created by @joeldhartman and more information can be found [here](https://www.astro.princeton.edu/~jhartman/vartools.html). The GitHub repo can also be found [here](https://github.com/joeldhartman/vartools)
