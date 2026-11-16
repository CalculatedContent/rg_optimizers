"""Publish allowlisted results using a clean worktree and a fresh main-head lease."""
import shutil, subprocess, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p/'.git').exists())
REL=ROOT.relative_to(REPO)
ALLOW=['results','plots','RESULTS.md']

def git(*args,cwd=REPO):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()

def main():
    if not (ROOT/'RESULTS.md').exists():
        print('No report exists; saved artifacts remain authoritative.')
        return
    git('fetch','origin','main')
    head=git('rev-parse','origin/main')
    dest=Path(tempfile.mkdtemp(prefix='fashion-adamw-paired120-publish-'))/'worktree'
    git('worktree','add','--detach',str(dest),head)
    try:
        target=dest/REL
        target.mkdir(parents=True,exist_ok=True)
        for name in ALLOW:
            src=ROOT/name
            if src.is_dir():shutil.copytree(src,target/name,dirs_exist_ok=True)
            elif src.exists():shutil.copy2(src,target/name)
        git('config','user.name','github-actions[bot]',cwd=dest)
        git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com',cwd=dest)
        for name in ALLOW:
            if (target/name).exists():git('add','-f',str(REL/name),cwd=dest)
        if not git('diff','--cached','--name-only',cwd=dest):
            print('Identical publication already exists; no duplicate commit.')
            return
        git('commit','-m','Report native AdamW paired paired120 Fashion trajectories with all outcomes',cwd=dest)
        git('push',f'--force-with-lease=refs/heads/main:{head}','origin','HEAD:main',cwd=dest)
        print('Published',git('rev-parse','HEAD',cwd=dest))
    finally:
        git('worktree','remove','--force',str(dest))

if __name__=='__main__':main()
