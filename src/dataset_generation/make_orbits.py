import sys
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import datetime
from lisaorbits import OEMOrbits, KeplerianOrbits, StaticConstellation
from helpers.config import *

ORBITS = OEMOrbits.from_included('esa-trailing')
YEARS = 2
DT = 1e5

def make_oem_orbits(years_commissioning = YEARS, orbits_dt = DT):
    orbits_t0 = (
        ORBITS.t_start + datetime.timedelta(days=years_commissioning * 365).total_seconds()
    )

    orbits_duration = datetime.timedelta(
        days=years_commissioning * 365
    ).total_seconds() 

    orbits_size = int(orbits_duration / orbits_dt)

    ORBITS.write(orbits_path, dt=orbits_dt, size=orbits_size, t0=orbits_t0, mode="w")

def main():
    make_oem_orbits()

if __name__ == "__main__":
    main()