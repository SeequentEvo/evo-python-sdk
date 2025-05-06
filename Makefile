lint:
	uv run --only-dev ruff check
	uv run --only-dev ruff format --check

lint-fix:
	uv run --only-dev ruff format

test-common:
	uv run --package evo-sdk-common pytest packages/evo-sdk-common/tests

test-files:
	uv run --package evo-files pytest packages/evo-files/tests

test-objects:
	uv run --package evo-objects pytest packages/evo-objects/tests

test:
	@make test-common
	@make test-files
	@make test-objects
