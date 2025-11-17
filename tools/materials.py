# materials.py
# Simple mapping of friendly names -> meent material table keys (if present).
# Add or edit entries according to your meent material table.

MAPPING = {
    "Si":    "p_Si__real",
    "SiO2":  "SiO2__real",
    "Si3N4": "Si3N4__real",   # <-- replace with correct key if different
    "Air":   None,             # None -> refractive index = 1.0
    # add more friendly names mapping to meent keys here...
}

# Fallback numeric refractive indices you want to use if material key not found:
FALLBACK_N = {
    "Si": 3.5,
    "SiO2": 1.45,
    "Si3N4": 2.0,
    "Air": 1.0,
}
