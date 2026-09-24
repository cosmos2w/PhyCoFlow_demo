# MIMONet source

- Source: Zenodo v2, https://zenodo.org/records/21986357 (DOI 10.5281/zenodo.21986357), retrieved 2026-09-24. The release metadata links https://github.com/kkazuma19/MIMONet; that repository returned HTTP 404 on retrieval, so no Git commit is available.
- Adapted files: `src/mimonet.py` and `src/fcn.py` from `MIMONet.zip`. The source edits are a package-relative `FCN` import and removal of trailing whitespace. The combustion data and training wrapper live outside this directory.
- License: the Zenodo v2 metadata says the code is MIT licensed, while the 3.7 GB ZIP contains no LICENSE file. `LICENSE` here records the stated MIT grant and credits the released authors. No upstream datasets were copied.
- Released example: `LDC/train.py` uses latent/basis dimension 256, branch width 512, trunk width 256, ReLU, multiplicative merge and MSE. The wrapper retains these dimensions and changes branch input sizes and output count for the combustion task.
