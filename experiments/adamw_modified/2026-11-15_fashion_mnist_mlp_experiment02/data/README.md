# Fashion-MNIST data

No data have been downloaded or copied into this folder by the setup task.

Use the four Fashion-MNIST IDX gzip files from the [official dataset repository](https://github.com/zalandoresearch/fashion-mnist#get-the-data). Their filenames match MNIST, so filename checks alone are insufficient. Verify the published MD5 checksums in `protocol/Protocol_Plan.json` before loading, then record SHA256 hashes for the execution freeze.

| File | Official MD5 |
|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

Recover `split.npz` from the [original saved MNIST artifacts](../../2026-11-13_mnist_mlp_experiment01/ARTIFACTS.md). Reuse its indices, not its images. Required SHA256: `ad533c8cb7a03eeb6c0a69b23e51eb074a273ff42f80bce229aca381f758d543`. The split selects 55,000 training and 5,000 validation examples from Fashion-MNIST's training set; its 10,000 test examples remain separate.

Flatten to 784 float32 pixels and divide by 255, matching Experiment 01. Do not add augmentation or substitute digit MNIST. Large dataset files belong in the experiment's saved artifact bundle; do not copy them into the original experiment folder.
