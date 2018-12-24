import os
from sarangi import main

if 'STRING_SIM_ROOT' not in os.environ:
    os.environ['STRING_SIM_ROOT'] = os.getcwd()

main()