try:
    from .ui_srrf import *

    main()
except Exception as e:
    print(e)
    from .srrf import *

    main()
