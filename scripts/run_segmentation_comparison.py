"""Run frozen recipes sequentially, one fresh process per model for memory isolation."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for recipe in ('baseline','box_prompt','direct_l','direct_x'):
        print(f'Running {recipe}',flush=True)
        with (args.output/f'{recipe}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','training.segmentation.evaluate','run',
                '--protocol',str(args.protocol),'--recipe',recipe,'--output',str(args.output/recipe)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Finished {recipe}',flush=True)


if __name__=='__main__': main()
