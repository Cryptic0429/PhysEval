#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    argv = sys.argv[1:]
    if "--auto-init-num-objects" not in argv:
        argv.extend(["--auto-init-num-objects", "2"])
    if "--auto-init" not in argv:
        argv.append("--auto-init")

    sys.argv = [sys.argv[0]] + argv

    from run_sam2_track import main as run_main

    run_main()


if __name__ == "__main__":
    main()
