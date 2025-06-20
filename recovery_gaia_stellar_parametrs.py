#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:18:36 2024

@author: nbadolo
"""

import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


#Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(0.1, u.deg)
height = u.Quantity(0.1, u.deg)
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
#%%
r.pprint(max_lines=12, max_width=130)
