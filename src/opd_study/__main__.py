"""Allow ``python -m opd_study`` to invoke the CLI."""

from opd_study.cli import main

raise SystemExit(main())
