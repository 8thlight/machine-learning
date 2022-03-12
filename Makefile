test: 
	pytest

lint-check:
	pylint src

lint-fix:
	autopep8

install:
	pipenv install

clean:
	pipenv clean