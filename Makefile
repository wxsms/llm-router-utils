.PHONY: help install test test-unit test-integration build clean lint check-torch

help:
	@echo "llm-router-utils — common commands"
	@echo ""
	@echo "  make install        Editable install with test extras"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only (migrated from upstream sglang)"
	@echo "  make test-integration  Run _process_messages integration tests"
	@echo "  make build          Build wheel + sdist into dist/"
	@echo "  make check-torch    Verify no direct 'import torch' in src/"
	@echo "  make clean          Remove build/dist/cache artifacts"

install:
	python -m pip install -e ".[test]"

test:
	PYTHONPATH=src python -m pytest test/

test-unit:
	PYTHONPATH=src python -m pytest test/unit/

test-integration:
	PYTHONPATH=src python -m pytest test/integration/

build:
	python -m pip install --quiet build
	python -m build

check-torch:
	@echo "Checking for direct torch imports in src/..."
	@if grep -rn "^import torch\|^from torch" src/ 2>/dev/null | grep -v __pycache__; then \
		echo "FAIL: direct torch import found (see above)"; exit 1; \
	else \
		echo "OK: no direct torch import in src/"; \
	fi

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
