Testing scripts locally:

1. Install `harmonica` (for example, editable installation):

```bash
$ pip install -e .
```

2. Launch a "fake" EPICS server in a separate terminal (requires `epics-base`):

```bash
$ cd elettra && softIoc -d harmonica.db
```

3. Load data into EPICS:
```bash
$ ./load.py -tn --sx=0.000025 --sy=0.000025 -f reference.pkl.gz
```

- `reference.pkl.gz` contains TbT data for lattice without errors
- `tbt.pkl.gz` contains TbT data for lattice with errors (around 10-20% rms beta beating)

4. Navigate to the `script` directory and open `test.ipynb`, or execute scripts separately.
