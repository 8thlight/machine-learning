# Machine Learning
Machine Learning tools, techniques, gists and projects. Some of this code is referenced in our Blog.

This repository uses `pipenv` as an environment manager.
The base python version is `3.9`. 

## Install dependencies
You will need a base `python` installed in your system.
```bash
python --version
```

Then you will need to install `pip` and `pipenv`.
```bash
python -m pip install --user pip --upgrade pip
python -m pip install --user pipenv
```

Install the dependencies with 
```bash
make install
```
This calls `pipenv`, which will create a virtual environment for this project.

To activate this environment run
```bash
make activate
```

## Tests
Run the entire test suite with `pytest`.
Use
```bash
make test
```

## Code Style
We use PEP8 as a style guide for python code.

Check lint errors with
```bash
make lint
```

We use `autopep8` to automatically fix errors.

Use
```bash
make lintfix
```
or
```bash
make lintfixhard
```

for in-place fixing of lint errors under the `/src` dir.


## Contribution
1. Create a new feature branch that compares to the main branch and open a PR.
1. Ensure you have written appropriate tests and they are passing.
1. Ensure the code passes the style guide conventions.

Update the `requirements.txt` file using the following command from the 
main directory:
```bash
make lock
```