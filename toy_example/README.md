## Toy model for string method

make initial "trajectory":

`python setup/toymodel.py --init --output traj.dcd`


import into sarangi:

`bow import --branch AZ --fields X Y --command "python setup/colvars.py" traj.dcd`

(to demonstate how difficult it is to work in full-dimensional space, omit the argument --fields X Y)

(to demonstrate how the algorithm can pick up Y automatically, use --fields X)

run string iterations:

`bow run --branch AZ --local --wait`

