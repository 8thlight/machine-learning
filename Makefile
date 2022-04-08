# Usage:
# make           # setup the project for the first time
# make install   # install pipenv dependencies
# make packages  # adds src/ files to be available as python modules

# default target
.DEFAULT_GOAL := init

# targets that do not create a file
.PHONY: activate test lint lint-src lintfix lintfixhard install lock clean

PY_FILES := src/ cli/

init: activate packages install test

activate:
	pipenv shell

test: 
	pytest

lint:
	pylint $(PY_FILES)

lintfix:
	autopep8 $(PY_FILES) --recursive --in-place --aggressive

lintfix-hard:
	autopep8 $(PY_FILES) --recursive --in-place --aggressive --aggressive

install:
	pipenv install --dev

lock:
	pipenv lock

clean:
	pipenv clean

packages:
	python path_adder.py
