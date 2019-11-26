set -e
#rm velout* out* init* || true

# adjust path to you namd2 executable here
mynamd=$HOME/opt/NAMD_git/namd2

if [ -z "$STRING_SIM_ROOT" ]
then
  # for testing only
  export STRING_SIM_ROOT=$HOME/git/sarangi/deca_ala
fi

export STAGE=equilibration
$mynamd +p 6 production_swarm.inp >> equilibration.log 2>&1

final_frames=""
final_colvars=""
final_vels=""
export STAGE=propagation
for i in {1..100}
do
  export i
  $mynamd +p 6 production_swarm.inp >> propagation.log 2>&1
  final_frames="$final_frames out_$i.dcd"
  final_colvars="$final_colvars out_$i.colvars.traj"
  final_vels="$final_frames velout_$i.dcd"
done

# concatenate results
mdconvert -f -o out.dcd $final_frames
cat $final_colvars > out.colvars.traj; # That's not particularly useful, since the step column will be messed up. Better recompute colvars
#mdconvert -f -o velout.dcd $final_vels
