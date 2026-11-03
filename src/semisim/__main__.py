"""Allow ``python -m semisim``."""

import sys

from .cli import main

sys.exit(main())
