from astropy.table import Table
import numpy as np
import re

table_path = '/home/nbadolo/Bureau/Aymard/These/CDS-Tables'


# ============================
# CLEAN UTILITIES (CDS SAFE)
# ============================

def extract_float(x):
    """Extract first valid float from messy CDS strings"""
    if x is None:
        return np.nan

    s = str(x).replace(",", ".")
    s = re.sub(r"[^0-9eE\+\-\.]", " ", s)

    match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if not match:
        return np.nan

    try:
        return float(match[0])
    except:
        return np.nan


def clean_numeric(table, col):
    if col in table.colnames:
        table[col] = [extract_float(x) for x in table[col]]


def clean_string(table, col):
    if col in table.colnames:
        table[col] = [str(x).strip() for x in table[col]]


# ==============================
# 1. TABLE 3
# ==============================

t3 = Table.read(f"{table_path}/table3_clean.dat", format="ascii.fixed_width_no_header")

# Assign column names (based on original table3 structure)
col_names_t3 = ["STAR", "STELLAR_TYPE", "DIST_PC", "VAR_TYPE", "PERIOD_D", "TEFF_K", "LUM", "MDOT_MSUNYR", "MAX_DOLP_PCT", "ROUT_RSTAR", "COMPANION"]
for i, name in enumerate(col_names_t3):
    if i < len(t3.colnames):
        t3.rename_column(t3.colnames[i], name)

num_cols_t3 = [
    "DIST_PC", "PERIOD_D", "TEFF_K", "LUM",
    "MDOT_MSUNYR", "MAX_DOLP_PCT", "ROUT_RSTAR"
]

for c in num_cols_t3:
    clean_numeric(t3, c)

for c in t3.colnames:
    if c not in num_cols_t3:
        clean_string(t3, c)

t3.write(f"{table_path}/table3.csv", format="ascii.csv", overwrite=True)


# ==============================
# 2. TABLE B1
# ==============================

tB1 = Table.read(f"{table_path}/tableB1_clean.dat", format="ascii.fixed_width_no_header")

# Assign column names (based on original tableB1 structure)
col_names_b1 = ["NAME", "LUM", "DIST_PC", "AV_MAG", "EIR", "LIR_LSTAR", "CTR", "RSTAR_MAS", "MV", "MJ", "MK", "JK", "BA", "E", "EPSILON", "RES_STAT"]
for i, name in enumerate(col_names_b1):
    if i < len(tB1.colnames):
        tB1.rename_column(tB1.colnames[i], name)

num_cols_b1 = [
    "LUM", "DIST_PC", "AV_MAG", "EIR",
    "LIR_LSTAR", "CTR", "RSTAR_MAS",
    "MV", "MJ", "MK", "JK",
    "BA", "E", "EPSILON"
]

for c in num_cols_b1:
    clean_numeric(tB1, c)

for c in tB1.colnames:
    if c not in num_cols_b1:
        clean_string(tB1, c)

tB1.write(f"{table_path}/tableB1.csv", format="ascii.csv", overwrite=True)


# ==============================
# 3. TABLE B2
# ==============================

tB2 = Table.read(f"{table_path}/tableB2_clean.dat", format="ascii.fixed_width_no_header")

# Assign column names (based on original tableB2 structure)
col_names_b2 = ["DATE", "TIME", "TARGET", "PROGRAM_ID", "DEDICATED_PSF", "FILTER1", "FILTER2", "SEEING", "AIRMASS"]
for i, name in enumerate(col_names_b2):
    if i < len(tB2.colnames):
        tB2.rename_column(tB2.colnames[i], name)

for c in ["SEEING", "AIRMASS"]:
    clean_numeric(tB2, c)

for c in tB2.colnames:
    if c not in ["SEEING", "AIRMASS"]:
        clean_string(tB2, c)

tB2.write(f"{table_path}/tableB2.csv", format="ascii.csv", overwrite=True)


# ==============================
# FINAL CHECK
# ==============================

print("CSV files generated for VizieR ✔")
print(f"-> {table_path}/table3.csv")
print(f"-> {table_path}/tableB1.csv")
print(f"-> {table_path}/tableB2.csv")