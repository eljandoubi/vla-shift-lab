"""Turn the result directories of a Colab run into tables and figures.

python scripts/make_tables.py --runs runs --out results [--readme README.md]
"""

import argparse
from pathlib import Path

from vlashift.report import update_readme, write_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="<arm>/<suite> tree")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, help="replace the results block of this file")
    args = parser.parse_args()
    table = write_all(args.runs, args.out)
    if args.readme:
        update_readme(args.readme, table)
    print(table)


if __name__ == "__main__":
    main()
