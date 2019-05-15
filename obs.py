import sys
from sarangi.observables import main_update

if len(sys.argv) > 1:
    main_update(image_id=sys.argv[1])
else:
    main_update()
