[![Build Status](https://app.travis-ci.com/8thlight/machine-learning.svg?branch=main)](https://app.travis-ci.com/8thlight/machine-learning)

# Machine Learning
Machine Learning tools, techniques, gists and projects.
Some of this code is referenced in our Blog.

![ezgif-3-53a018627a](https://user-images.githubusercontent.com/25442086/168663740-ea7ebc04-71fa-4800-a0ff-6eb5b389c5c7.gif)

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

To add this project's packages to the environment run
```bash
make packages
```

Install and lock packages always through `make`, as it will handle different platforms for you.
If a new platform is not supported, please update the `platform_pipfile.py` appropriately, or create a new platform directory if necessary. This should be a temporary fix, as Pipenv creates multi-platform Pipfile.lock [support](https://github.com/pypa/pipenv/issues/5130).

To see graphs of ML models using `tf.keras.utils.plot_model`, you will also need [graphviz](https://graphviz.gitlab.io/download/).

The `[packages]` and `[dev-packages]` sections of the Pipfiles correspond to dependencies imported in the code, and those used through the cli respectively.

## Running examples
Every file in the `cli/` folder is an independent example available through
CLI commands. Use `python <file>.py --help` to see the available options
for the given example.

For example, try running the Snake Game:
```shell
.../cli > python play_snake.py
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

## VSCode
If you are using VSCode, the virtual environment created by `pipenv` will not 
be immediately available and you will see warnings in your import statements.
To fix this first make sure the appropriate virtual environment is activated
by running `make activate`, then get the location of the current python
interpreter using `make python`. The printed line should look something like
this:

```bash
/Users/yourname/path/to/virtualenvs/machine-learning-abcde1234/bin/python
```

Copy that line. Then open your
[settings.json](https://code.visualstudio.com/docs/getstarted/settings)
file and add a new key `"python.defaultInterpreterPath"`, then paste the 
previously copied python interpreter path as its value and restart VSCode.

```json
{
    "python.defaultInterpreterPath": "/Users/yourname/path/to/virtualenvs/machine-learning-abcde1234/bin/python"
}
```
## Contribution
1. Create a new feature branch that compares to the main branch and open a PR.
1. Ensure you have written appropriate tests and they are passing.
1. Ensure the code passes the style guide conventions.

If some dependencies were added, update the `Pipfile.lock` file
using:
```bash
make lock
```
