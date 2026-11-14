PYTHON ?= python
CONFIG ?= config.json

.PHONY: help install install-dev install-server upgrade-pillow setup-precommit \
	init init-full run gui formats platforms download-only server \
	build appimage snap \
	lint format format-check mypy vulture deptry \
	test test-no-cov test-focused test-config test-app test-network test-e2e test-ladder \
	test-memory test-all \
	check

install:
	pip install -r requirements-dev.txt

install-dev: install
	pip install -e .

install-server:
	pip install -e ".[server]"

setup-precommit:
	pre-commit install

init:
	$(PYTHON) -m autoclip --init $(CONFIG)

init-full:
	$(PYTHON) -m autoclip --init-full $(CONFIG)

run:
	$(PYTHON) -m autoclip $(CONFIG)

formats:
	$(PYTHON) -m autoclip --formats $(CONFIG)

platforms:
	$(PYTHON) -m autoclip --platforms $(CONFIG)

download-only:
	$(PYTHON) -m autoclip --download-only $(CONFIG)

gui:
	$(PYTHON) -m autoclip --gui

server:
	$(PYTHON) -m autoclip.server

build:
	$(PYTHON) -m PyInstaller autoclip.spec --noconfirm

appimage: build
	$(PYTHON) tools/make_appimage.py

snap: build
	sudo snapcraft --destructive-mode

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

mypy:
	mypy autoclip tests tools

vulture:
	vulture

deptry:
	deptry .

test:
	pytest

test-no-cov:
	pytest --no-cov

test-focused:
	$(PYTHON) -m pytest --no-cov

test-config:
	$(PYTHON) -m pytest tests/unit/test_config.py --no-cov

test-app:
	$(PYTHON) -m pytest tests/unit/test_app.py tests/unit/test_session_direct.py --no-cov

test-network:
	AUTOCLIP_NETWORK_TESTS=1 $(PYTHON) -m pytest -m "network and not e2e" --no-cov

test-e2e:
	AUTOCLIP_E2E_TESTS=1 $(PYTHON) -m pytest tests/e2e -m e2e --no-cov

test-ladder:
	AUTOCLIP_E2E_TESTS=1 AUTOCLIP_E2E_LEVEL=3 $(PYTHON) -m pytest tests/e2e -m e2e --no-cov

test-memory:
	AUTOCLIP_MEMORY_TESTS=1 $(PYTHON) -m pytest tests/memory --no-cov

check: mypy vulture deptry lint format-check test-all test-memory

test-all:
	AUTOCLIP_NETWORK_TESTS=1 AUTOCLIP_E2E_TESTS=1 AUTOCLIP_E2E_LEVEL=3 $(PYTHON) -m pytest
