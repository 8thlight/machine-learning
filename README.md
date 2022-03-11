# Machine Learning
Machine Learning tools, techniques, gists and projects. Some of this code is referenced in our Blog.

This repository uses `conda` as an environment manager.
The base python version is `3.9.7`. 

## Install dependencies
You can follow [this instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
to install conda.

Once conda is installed run the following command to install all dependencies
into a separate environment called `machine-learning`.
```bash
conda create --name machine-learning --file requirements.txt
```

## Tests
Run the entire test suite with `pytest`.



## Contribution
1. Create a new feature branch that compares to the main branch and open a PR.
1. Ensure you have written appropriate tests and run `pytest`

Update the `requirements.txt` file using the following command from the 
main directory:
```bash
conda list -e > requirements.txt
```