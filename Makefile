activate:
	pipenv shell

test: 
	pytest

lint:
	pylint **/*.py

lint-src:
	pylint src/

lintfix:
	autopep8 **/*.py --recursive --in-place --aggressive

lintfixhard:
	autopep8 **/*.py --recursive --in-place --aggressive --aggressive

install:
	pipenv install --dev

lock:
	pipenv lock

clean:
	pipenv clean
