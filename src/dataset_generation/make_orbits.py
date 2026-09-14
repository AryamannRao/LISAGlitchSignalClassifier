# Create the precomputed LISA orbit file consumed by the dataset generators.
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
# Generate four years of orbit samples at the configured cadence.
YEARS = 4
DT = 1e5

def make_oem_orbits(years_commissioning = YEARS, orbits_dt = DT):
    # Start after commissioning and write an OEM orbit ephemeris to disk.
    orbits_t0 = (
        ORBITS.t_start + datetime.timedelta(days=years_commissioning * 365).total_seconds()
    )

    orbits_duration = datetime.timedelta(
        days=years_commissioning * 365
    ).total_seconds() 

    # Convert the requested duration and cadence into the required sample count.
    orbits_size = int(orbits_duration / orbits_dt)

    ORBITS.write(orbits_path, dt=orbits_dt, size=orbits_size, t0=orbits_t0, mode="w")

def main():
    # Use the module defaults to generate the project orbit file.
    make_oem_orbits()

if __name__ == "__main__":
    main()
