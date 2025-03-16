init:
	cd backend && \
	uv venv && \
	source .venv/bin/activate && uv sync

lint:
	uv run ruff check backend

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
