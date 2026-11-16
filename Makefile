lint:
	uv run --only-dev ruff check
	uv run --only-dev ruff format --check

lint-fix:
	uv run --only-dev ruff check --fix
	uv run --only-dev ruff format

stubs-compute:
	uv run --package evo-compute python -m evo.compute._stubgen generate

check-stubs-compute:
	uv run --package evo-compute python -m evo.compute._stubgen generate --check

check-types-compute:
	uv run --package evo-compute --group test pytest packages/evo-compute/tests/test_stubgen.py::TestTypeCheckers -q

test-common:
	uv run --package evo-sdk-common pytest packages/evo-sdk-common/tests

test-blockmodels:
	uv run --package evo-blockmodels pytest packages/evo-blockmodels/tests

test-files:
	uv run --package evo-files pytest packages/evo-files/tests

test-objects:
	uv run --package evo-objects pytest packages/evo-objects/tests

test-colormaps:
	uv run --package evo-colormaps pytest packages/evo-colormaps/tests

test-widgets:
	uv run --package evo-widgets pytest packages/evo-widgets/tests

test-notebooks:
	uv run --directory code-samples --group test pytest tests/ -v

test:
	@make test-common
	@make test-blockmodels
	@make test-files
	@make test-objects
	@make test-colormaps
	@make test-widgets
