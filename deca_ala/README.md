
run pulling simulation

`( cd setup; bash pulling.sh )`

Create initial string with sarangi's import command:

`bow import --fields endtoend RMSDtofolded --max_images inf --branch AZ --command 'python $STRING_SIM_ROOT/setup/colvars.py' pulling/out.dcd`

