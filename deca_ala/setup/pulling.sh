set -e

export STRING_SIM_ROOT=/home/fab/git/sarangi/deca_ala/

cd $STRING_SIM_ROOT/pulling

~/opt/NAMD/namd2 +p 12 $STRING_SIM_ROOT/setup/pulling.namd

