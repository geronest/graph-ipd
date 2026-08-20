"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

run train.py multiple times with configurations
specified in a experiment set.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="arguments for running a set of experiments"
    )
    # determine which configuration to use
    parser.add_argument("set_id", type=str)
    parser.add_argument("run_id", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    l_configs = os.listdir(f"./configs/{args.set_id}")

    for cfg in l_configs:
        cmd = (
            f"python3 bin/train.py {cfg[:-5]}"
            + f" --set_id {args.set_id} --run_id {args.run_id}"
        )
        os.system(cmd)
