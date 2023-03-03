# default target
.DEFAULT_GOAL := init

# targets that do not create a file
.PHONY: init activate test lint lint-src lintfix lintfixhard install lock clean

# help target taken from https://gist.github.com/prwhite/8168133
help: ## Shows help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% 0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

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

pipfile: install

install:
	@PLATFORM=$$(python platform_pipfile.py) && \
	cp platforms/$$PLATFORM/Pipfile Pipfile && \
	cp platforms/$$PLATFORM/Pipfile.lock Pipfile.lock && \
	echo "Installing for platform $$PLATFORM" && \
	pipenv install --dev;

mmm:
	cp Pipfile platforms/arm64/Pipfile; \

lock:
	@PLATFORM=$$(python platform_pipfile.py) && \
	pipenv lock && \
	cp Pipfile platforms/$$PLATFORM/Pipfile && \
	cp Pipfile.lock platforms/$$PLATFORM/Pipfile.lock && \
	echo "Locking for platform $$PLATFORM";

clean:
	pipenv clean

packages:
	python path_adder.py
