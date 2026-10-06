"""Install the XLA 2.6 Pallas extras into a per-run overlay, not the shared venv.

Version authority: pytorch/xla v2.6.0 setup.py extras_require['pallas'].
JAX traces TPU kernels; PyTorch/XLA remains the training runtime.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = ('jax==0.4.38', 'jaxlib==0.4.38', 'ml-dtypes==0.5.1', 'opt-einsum==3.4.0')
CORE = ('torch', 'torch-xla', 'libtpu', 'numpy', 'scipy')


def versions():
    result = {}
    for name in CORE:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def install_command(target):
    return [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--no-deps', '--only-binary=:all:', '--timeout', '120', '--retries', '3',
            '--target', str(target), *PACKAGES]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    before = versions()
    if not (before['torch-xla'] or '').startswith('2.6.'):
        raise RuntimeError('These Pallas pins require the installed PyTorch/XLA 2.6 runtime')
    # The overlay intentionally reuses existing NumPy/SciPy rather than upgrading
    # their binary dependencies in the known training environment.
    from packaging.version import Version
    for name, floor in (('numpy','1.24'),('scipy','1.10')):
        if before[name] is None or Version(before[name]) < Version(floor):
            raise RuntimeError(name+' does not satisfy JAX 0.4.38; no shared packages changed')
    target = a.root/'pallas-deps'
    subprocess.run(install_command(target), check=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(target)+os.pathsep+env.get('PYTHONPATH','')
    # Import verification only, on CPU. Training inherits no JAX_PLATFORMS override;
    # Pallas must lower TPU code and uses XLA's jax_import_guard() for device ownership.
    env['JAX_PLATFORMS'] = 'cpu'
    code = '''import json, jax, jaxlib, ml_dtypes, opt_einsum
from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl, _flash_attention_bwd_dq, _flash_attention_bwd_dkv
assert jax.__version__ == jaxlib.__version__ == "0.4.38"
print(json.dumps({"jax":jax.__version__, "jaxlib":jaxlib.__version__, "ml_dtypes":ml_dtypes.__version__, "opt_einsum":opt_einsum.__version__}))'''
    result = subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True)
    after = versions()
    if after != before:
        raise RuntimeError('Unexpected change to shared core-package versions')
    record = {'status':'installed_and_import_verified','packages':json.loads(result.stdout),
              'overlay':str(target),'core_before':before,'core_after':after,
              'source':'https://github.com/pytorch/xla/blob/v2.6.0/setup.py'}
    (a.root/'PALLAS_DEPENDENCIES.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
