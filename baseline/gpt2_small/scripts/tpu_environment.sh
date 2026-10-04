# Source before importing torch_xla. Matches the verified v5litepod-8 allocation.
export PJRT_DEVICE=TPU TPU_ACCELERATOR_TYPE=v5litepod-8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if systemctl is-active --quiet rg-continuous8.service; then
  echo 'Old training service is still active; refusing concurrent TPU use.' >&2
  exit 1
fi
# The previous service ran as root; libtpu writes here even for a user-run job.
if [ -L /tmp/tpu_logs ]; then
  echo 'Unexpected TPU log directory symlink.' >&2
  exit 1
fi
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" /tmp/tpu_logs
test -w /tmp/tpu_logs
