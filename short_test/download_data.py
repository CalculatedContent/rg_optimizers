"""Fetch the public input and require the exact pilot SHA256."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
EXPECTED='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('input.txt'));a=p.parse_args()
    if a.output.exists():
        assert hashlib.sha256(a.output.read_bytes()).hexdigest()==EXPECTED
        print(a.output);return
    data=urlopen('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',timeout=60).read()
    assert hashlib.sha256(data).hexdigest()==EXPECTED
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(data);print(a.output)
if __name__=='__main__':main()
