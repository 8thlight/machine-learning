activate:
	pipenv shell

test: 
	pytest

lint:
	pylint **/*.py

lintfix:
	autopep8 **/*.py --recursive --in-place --aggressive

lintfixhard:
	autopep8 **/*.py --recursive --in-place --aggressive --aggressive

install:
	pipenv install

lock:
	pipenv lock

clean:
	pipenv clean