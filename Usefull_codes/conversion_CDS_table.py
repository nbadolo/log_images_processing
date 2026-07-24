from astropy.table import Table
from astropy.io import fits
import numpy as np

table_path = f'/home/nbadolo/Bureau/Aymard/These/CDS-Tables'

# ==============================
# 1. LECTURE DE LA TABLE ASCII
# ==============================

t = Table.read(f"{table_path}/table3.dat", format="ascii", delimiter="|")

# ==============================
# 2. NETTOYAGE DES VALEURS
# ==============================

def to_nan(val):
    if val in ["--", "-", "NaN", "nan", None]:
        return np.nan
    return val

for col in t.colnames:
    if t[col].dtype.kind in ['U', 'S']:
        t[col] = [to_nan(x) for x in t[col]]

# ================================
# 3. HEADER CDS / A&A (IMPORTANT)
# ================================

t.meta["TITLE"] = "Stellar, variability, mass-loss and polarimetric properties of 18 AGB stars observed with SPHERE/ZIMPOL"
t.meta["INSTRUME"] = "VLT/SPHERE-ZIMPOL"
t.meta["COMMENT"] = [
    "Catalog of stellar parameters and polarimetric properties",
    "for 18 resolved AGB stars observed in polarized visible light.",
]
t.meta["REFERENC"] = "Badolo et al. 2026 (A&A, accepted)"
t.meta["NOTE"] = (
    "MAX_DOLP_PCT and ROUT_RSTAR are derived from SPHERE/ZIMPOL "
    "polarimetric intensity and DoLP maps."
)

# ==============================
# 4. UNITÉS (CDS STANDARD)
# ==============================

t["DIST_PC"].unit = "pc"
t["PERIOD_D"].unit = "d"
t["TEFF_K"].unit = "K"
t["L_LSUN"].unit = "Lsun"
t["MDOT_MSUNYR"].unit = "Msun/yr"
t["MAX_DOLP_PCT"].unit = "%"
t["ROUT_RSTAR"].unit = "Rstar"

# ==============================
# 5. DESCRIPTIONS COLONNES (CDS IMPORTANT)
# ==============================

t["STAR"].description = "Name of the AGB star"
t["STELLAR_TYPE"].description = "Spectral classification"
t["DIST_PC"].description = "Distance from Earth"
t["VAR_TYPE"].description = "Variability type (Mira, SRa, SRb, etc.)"
t["PERIOD_D"].description = "Pulsation period"
t["TEFF_K"].description = "Effective temperature"
t["L_LSUN"].description = "Stellar luminosity"
t["MDOT_MSUNYR"].description = "Mass-loss rate"
t["MAX_DOLP_PCT"].description = "Maximum degree of linear polarization"
t["ROUT_RSTAR"].description = "Maximum extent of polarized dust emission"
t["COMPANION"].description = "Presence or separation of companion"

# ==============================
# 6. EXPORT FITS
# ==============================

t.write(f"{table_path}/table3_cds.fits", format="fits", overwrite=True)

print("CDS-ready FITS file created: table3_cds.fits")
# print(t)
# print(t.meta)
hdul = fits.open(f"{table_path}/table3_cds.fits")
hdul.info()
print(t.colnames)
t.pprint(max_width=-1, max_lines=-1)


# ==============================
# Table B.1. 
# ==============================

tB1 = Table.read(f"{table_path}/tableB1.dat", format="ascii", delimiter="|")

# -----------------------
# HEADER CDS
# -----------------------
tB1.meta["TITLE"] = "Stellar and photometric properties of AGB stars"
tB1.meta["INSTRUME"] = "Literature + SPHERE ancillary data"
tB1.meta["DESCRIPTION"] = "Physical and photometric parameters of AGB stars"
tB1.meta["REFERENC"] = "Badolo et al. A&A accepted"

# -----------------------
# UNITS (CDS STANDARD)
# -----------------------
tB1["L_LSUN"].unit = "Lsun"
tB1["D_PC"].unit = "pc"
tB1["AV_MAG"].unit = "mag"
tB1["MV"].unit = "mag"
tB1["MJ"].unit = "mag"
tB1["MK"].unit = "mag"


# dimensionless:
tB1["LIR_LSTAR"].unit = ""
tB1["CTR"].unit = ""
tB1["B_A"].unit = ""
tB1["E"].unit = ""
tB1["EPSILON"].unit = ""
tB1["EIR"].unit = ""

# angular:
tB1["RSTAR_MAS"].unit = "mas"

# ==============================
#  DESCRIPTIONS COLONNES (CDS IMPORTANT)
# ==============================
tB1["L_LSUN"].description = "Stellar luminosity"
tB1["D_PC"].description = "Distance"
tB1["AV_MAG"].description = "Visual extinction"
tB1["EIR"].description = "Infrared excess"
tB1["LIR_LSTAR"].description = "Infrared to stellar luminosity ratio"
tB1["CTR"].description = "Concentration index"
tB1["RSTAR_MAS"].description = "Angular stellar radius"
tB1["MV"].description = "Absolute V magnitude"
tB1["MJ"].description = "Absolute J magnitude"
tB1["MK"].description = "Absolute K magnitude"

# Ensure `JK` exists: compute from `MJ` and `MK` when possible, otherwise
# create a column of NaNs to avoid KeyError when setting metadata or writing.
if "JK" not in tB1.colnames:
    if ("MJ" in tB1.colnames) and ("MK" in tB1.colnames):
        try:
            tB1["MJ"] = tB1["MJ"].astype(float)
            tB1["MK"] = tB1["MK"].astype(float)
            tB1["JK"] = tB1["MJ"] - tB1["MK"]
        except Exception:
            tB1["JK"] = [np.nan] * len(tB1)
    else:
        tB1["JK"] = [np.nan] * len(tB1)
# Now safe to set `JK` unit and description
tB1["JK"].unit = "mag"
tB1["JK"].description = "J-K colour index"
tB1["B_A"].description = "Axial ratio"
tB1["E"].description = "Eccentricity"
tB1["EPSILON"].description = "Ellipticity"
tB1["RES_STAT"].description = "Envelope resolution status"

tB1.write(f"{table_path}/tableB1_cds.fits", format="fits", overwrite=True)

print("B.1 FITS CDS-ready ")
hdulB1 = fits.open(f"{table_path}/tableB1_cds.fits")
hdulB1.info()
print(tB1.colnames)
tB1.pprint(max_width=-1, max_lines=-1)


# ==============================
# Table B.2 
# ==============================

tB2 = Table.read(f"{table_path}/tableB2.dat", format="ascii", delimiter="|")

# -----------------------
# CDS META
# -----------------------
tB2.meta["TITLE"] = "SPHERE/ZIMPOL observing log of AGB stars"
tB2.meta["INSTRUME"] = "VLT/SPHERE-ZIMPOL"
tB2.meta["REFERENC"] = "Badolo et al. A&A accepted"

# -----------------------
# UNITS
# -----------------------
tB2["SEEING"].unit = "arcsec"
tB2["AIRMASS"].unit = ""

# -----------------------
# OPTIONAL: clean strings
# -----------------------
tB2["DEDICATED_PSF"] = tB2["DEDICATED_PSF"].astype(str)
tB2["TARGET"] = tB2["TARGET"].astype(str)

# ==============================
#  DESCRIPTIONS COLONNES (CDS IMPORTANT)
# ==============================
# Ensure expected columns exist to avoid KeyError when assigning descriptions
for _col in ["DATE", "TIME", "TARGET", "PROGRAM_ID", "DEDICATED_PSF",
             "FILTER1", "FILTER2", "SEEING", "AIRMASS"]:
    if _col not in tB2.colnames:
        tB2[_col] = [""] * len(tB2)

tB2["DATE"].description = "Observation date"
tB2["TIME"].description = "Observation UTC time"
tB2["TARGET"].description = "Observed target"
tB2["PROGRAM_ID"].description = "ESO observing program"
tB2["DEDICATED_PSF"].description = "Dedicated PSF star available"
tB2["FILTER1"].description = "First ZIMPOL filter"
tB2["FILTER2"].description = "Second ZIMPOL filter"
tB2["SEEING"].description = "Atmospheric seeing"
tB2["AIRMASS"].description = "Observation airmass"

# -----------------------
# WRITE FITS
# -----------------------
tB2.write(f"{table_path}/tableB2_cds.fits", format="fits", overwrite=True)

print("Table B.2 FITS CDS created")
print("B.1 FITS CDS-ready created")
hdulB1 = fits.open(f"{table_path}/tableB1_cds.fits")
hdulB1.info()
print(tB1.colnames)
tB1.pprint(max_width=-1, max_lines=-1)

print("B.2 FITS CDS-ready created")
hdulB2 = fits.open(f"{table_path}/tableB2_cds.fits")
hdulB2.info()
print(tB2.colnames)
tB2.pprint(max_width=-1, max_lines=-1)

#hdul = fits.open("table3_cds.fits")
print(repr(hdul[1].header))
print(repr(hdulB1[1].header))
print(repr(hdulB2[1].header))