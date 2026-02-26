Testing scripts locally:

1. Install `harmonica` (use a new virtual environment):

```bash
$ conda create -n hs python=3.12 -c conda-forge --override-channels -y
$ conda activate hs
$ git clone https://github.com/i-a-morozov/harmonica.git
$ cd harmonica
$ pip install -e ".[script]"
```

2. Navigate to `harmonica/elettra/script` and start `jupyter lab`.

3. Follow instructions for setting up a fake EPICS or Tango environment.

4. Run scripts directly in the directory or from `jupyter lab`.
