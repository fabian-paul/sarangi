set -e

export STRING_SIM_ROOT=$(pwd)
mkdir -p work
cd work

python ../../string_archive.py extract --id AZ_001_000_000 --plan ../strings/AZ_001/plan.yaml

touch out.colvars.traj out.dcd
python ../../string_archive.py store --id AZ_001_000_000
