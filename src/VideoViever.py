import argparse
import textwrap
from pathlib import Path

from Video import Video

parser = argparse.ArgumentParser(description=textwrap.dedent('''\
    View Video and Navigate through Frames manually.

'''))

parser.add_argument('in_filename', help='Input filename')
# parser.add_argument('out_filename', help='Output filename')

if __name__ == '__main__':
    args = parser.parse_args()
    video = Video(Path(args.in_filename).resolve())
    player = video.showVideo()
    player.run()
