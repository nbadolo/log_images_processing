from astropy.table import Table
from astropy.io import fits
import numpy as np

table_path = '/home/nbadolo/Bureau/Aymard/These/CDS-Tables'


# ==============================
# UTILITAIRES
# ==============================

def clean_nan(val):
    """Convertit les valeurs texte en vrais NaN"""
    if val in ["--", "-", "NaN", "nan", "None", ""]:
        return np.nan
    return val


def force_float_column(table, col):
    """Force une colonne en float propre"""
    if col in table.colnames:
        try:
            table[col] = np.array(table[col], dtype=float)
        except Exception:
            # fallback sécurisé
            table[col] = [np.nan if x in ["--", "-", "nan", "NaN", ""] else float(x) for x in table[col]]


def round_column(table, col, decimals=3):
    """Arrondi propre des floats pour éviter 1.869999999"""
    if col in table.colnames:
        try:
            table[col] = np.round(np.array(table[col], dtype=float), decimals)
        except Exception:
            pass


def ensure_column_alias(table, target, aliases=()):
    """Renomme la première colonne trouvée parmi les alias vers le nom cible."""
    if target in table.colnames:
        return
    for alias in aliases:
        if alias in table.colnames:
            table.rename_column(alias, target)
            return


def set_unit(table, col, unit):
    """Définit une unité seulement si la colonne existe."""
    if col in table.colnames:
        table[col].unit = unit


def set_description(table, col, description):
    """Définit une description seulement si la colonne existe."""
    if col in table.colnames:
        table[col].description = description


# ==============================
# 1. TABLE 3
# ==============================

t = Table.read(f"{table_path}/table3.dat", format="ascii", delimiter="|")

for col in t.colnames:
    if t[col].dtype.kind in ['U', 'S']:
        t[col] = [clean_nan(x) for x in t[col]]

# forcing numeric columns
num_cols = ["DIST_PC", "PERIOD_D", "TEFF_K", "L_LSUN", "MDOT_MSUNYR",
            "MAX_DOLP_PCT", "ROUT_RSTAR"]

ensure_column_alias(t, "DIST_PC", aliases=("D_PC", "DIST", "DISTANCE_PC", "col1"))
ensure_column_alias(t, "L_LSUN", aliases=("LUM", "LUMINOSITY"))

for c in num_cols:
    force_float_column(t, c)
    round_column(t, c, 3)

# META
t.meta["TITLE"] = "Stellar, variability, mass-loss and polarimetric properties of 18 AGB stars observed with SPHERE/ZIMPOL"
t.meta["INSTRUME"] = "VLT/SPHERE-ZIMPOL"
t.meta["COMMENT"] = [
    "Catalog of stellar parameters and polarimetric properties",
    "for 18 resolved AGB stars observed in polarized visible light.",
]
t.meta["REFERENC"] = "Badolo et al. 2026 (A&A, accepted)"
t.meta["NOTE"] = "MAX_DOLP_PCT and ROUT_RSTAR derived from SPHERE/ZIMPOL"

# UNITS
set_unit(t, "DIST_PC", "pc")
set_unit(t, "PERIOD_D", "d")
set_unit(t, "TEFF_K", "K")
set_unit(t, "L_LSUN", "")
set_unit(t, "MDOT_MSUNYR", "Msun/yr")
set_unit(t, "MAX_DOLP_PCT", "%")
set_unit(t, "ROUT_RSTAR", "")

# DESCRIPTIONS
set_description(t, "STAR", "Name of the AGB star")
set_description(t, "STELLAR_TYPE", "Spectral classification")
set_description(t, "DIST_PC", "Distance from Earth")
set_description(t, "VAR_TYPE", "Variability type")
set_description(t, "PERIOD_D", "Pulsation period")
set_description(t, "TEFF_K", "Effective temperature")
set_description(t, "L_LSUN", "Stellar luminosity")
set_description(t, "MDOT_MSUNYR", "Mass-loss rate")
set_description(t, "MAX_DOLP_PCT", "Max linear polarization")
set_description(t, "ROUT_RSTAR", "Dust emission extent in stellar radii")
set_description(t, "COMPANION", "Companion presence")

if "L_LSUN" in t.colnames:
    t.rename_column("L_LSUN", "LUM")

t.write(f"{table_path}/table3.fits", overwrite=True, format="fits")


# ==============================
# 2. TABLE B1
# ==============================

tB1 = Table.read(f"{table_path}/tableB1.dat", format="ascii", delimiter="|")

for col in tB1.colnames:
    if tB1[col].dtype.kind in ['U', 'S']:
        tB1[col] = [clean_nan(x) for x in tB1[col]]

num_cols_b1 = ["L_LSUN", "D_PC", "AV_MAG", "MV", "MJ", "MK",
               "LIR_LSTAR", "CTR", "B_A", "E", "EPSILON", "EIR"]

for c in num_cols_b1:
    force_float_column(tB1, c)
    round_column(tB1, c, 3)

# META
tB1.meta["TITLE"] = "Stellar and photometric properties of AGB stars"
tB1.meta["INSTRUME"] = "Literature + SPHERE ancillary data"
tB1.meta["REFERENC"] = "Badolo et al. A&A accepted"

# UNITS
ensure_column_alias(tB1, "DIST_PC", aliases=("D_PC",))
ensure_column_alias(tB1, "L_LSUN", aliases=("LUM",))
set_unit(tB1, "L_LSUN", "")
set_unit(tB1, "DIST_PC", "pc")
set_unit(tB1, "AV_MAG", "mag")
set_unit(tB1, "MV", "mag")
set_unit(tB1, "MJ", "mag")
set_unit(tB1, "MK", "mag")
set_unit(tB1, "RSTAR_MAS", "mas")

# DESCRIPTIONS (inchangées)
set_description(tB1, "L_LSUN", "Stellar luminosity")
set_description(tB1, "DIST_PC", "Distance")
set_description(tB1, "AV_MAG", "Visual extinction")
set_description(tB1, "EIR", "Infrared excess")
set_description(tB1, "LIR_LSTAR", "IR/stellar luminosity ratio")
set_description(tB1, "CTR", "Concentration index")
set_description(tB1, "RSTAR_MAS", "Angular stellar radius")
set_description(tB1, "MV", "Absolute V magnitude")
set_description(tB1, "MJ", "Absolute J magnitude")
set_description(tB1, "MK", "Absolute K magnitude")
set_description(tB1, "B_A", "Axial ratio")
set_description(tB1, "E", "Eccentricity")
set_description(tB1, "EPSILON", "Ellipticity")
set_description(tB1, "RES_STAT", "Envelope resolution status")

ensure_column_alias(tB1, "JK", aliases=("J_K",))
ensure_column_alias(tB1, "BA", aliases=("B_A",))

if "L_LSUN" in tB1.colnames:
    tB1.rename_column("L_LSUN", "LUM")

tB1.write(f"{table_path}/tableB1.fits", overwrite=True, format="fits")


# ==============================
# 3. TABLE B2
# ==============================

tB2 = Table.read(f"{table_path}/tableB2.dat", format="ascii", delimiter="|")

for col in tB2.colnames:
    if tB2[col].dtype.kind in ['U', 'S']:
        tB2[col] = [clean_nan(x) for x in tB2[col]]

# forcing numeric
force_float_column(tB2, "SEEING")
force_float_column(tB2, "AIRMASS")

# META
tB2.meta["TITLE"] = "SPHERE/ZIMPOL observing log of AGB stars"
tB2.meta["INSTRUME"] = "VLT/SPHERE-ZIMPOL"
tB2.meta["REFERENC"] = "Badolo et al. A&A accepted"

# UNIT
# Ensure expected columns exist before assigning descriptions
for col in ["DATE", "TIME", "TARGET", "PROGRAM_ID", "DEDICATED_PSF", "FILTER1", "FILTER2", "SEEING", "AIRMASS"]:
    if col not in tB2.colnames:
        tB2[col] = [""] * len(tB2)

tB2["SEEING"].unit = "arcsec"

# DESCRIPTIONS
tB2["DATE"].description = "Observation date"
tB2["TIME"].description = "UTC time"
tB2["TARGET"].description = "Observed star"
tB2["PROGRAM_ID"].description = "ESO program"
tB2["DEDICATED_PSF"].description = "PSF availability"
tB2["FILTER1"].description = "Filter 1"
tB2["FILTER2"].description = "Filter 2"
tB2["SEEING"].description = "Atmospheric seeing"
tB2["AIRMASS"].description = "Airmass"

tB2.write(f"{table_path}/tableB2.fits", overwrite=True, format="fits")


# ==============================
# CHECK FITS
# ==============================

print("TABLE 3 OK")
hdul = fits.open(f"{table_path}/table3.fits")
hdul.info()
print(repr(hdul[1].header))
print("TABLE 3 DATA")
t.pprint(max_width=-1, max_lines=-1)

print("TABLE B1 OK")
hdulB1 = fits.open(f"{table_path}/tableB1.fits")
hdulB1.info()
print(repr(hdulB1[1].header))
print("TABLE B1 DATA")
tB1.pprint(max_width=-1, max_lines=-1)

print("TABLE B2 OK")
hdulB2 = fits.open(f"{table_path}/tableB2.fits")
hdulB2.info()
print(repr(hdulB2[1].header))
print("TABLE B2 DATA")
tB2.pprint(max_width=-1, max_lines=-1)

for c in tB1.colnames:
    print(c)


h = fits.open(f"{table_path}/table3.fits")
print(h[1].columns)


print("\nTABLE3")
print(fits.open(f"{table_path}/table3.fits")[1].columns)

print("\nTABLEB1")
print(fits.open(f"{table_path}/tableB1.fits")[1].columns)