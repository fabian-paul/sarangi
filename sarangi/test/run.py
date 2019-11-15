import os
from sarangi import main

if 'STRING_SIM_ROOT' not in os.environ:  # TODO: this is bit of a dirty hack, remove asap
    os.environ['STRING_SIM_ROOT'] = os.getcwd()

main()