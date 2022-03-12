test: 
	pytest

lint:
	pylint src

lintfix:
	autopep8 src --recursive --in-place --aggressive

lintfixhard:
	autopep8 src --recursive --in-place --aggressive --aggressive

install:
	pipenv install

lock:
	pipenv lock

clean:
	pipenv clean