init:
	cd backend && \
	uv venv && \
	source .venv/bin/activate && uv sync

lint:
	uv run ruff check backend

docker-build-api:
	docker build -t sports-pitch-api ./api

docker-build-frontend:
	docker build -t sports-pitch-frontend ./frontend

docker-build: docker-build-api docker-build-frontend
	@echo "All Docker images built successfully"

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

setup-minio:
	chmod +x ./scripts/setup_local_minio.sh
	./scripts/setup_local_minio.sh

local-dev:
	docker-compose up -d
	@echo "Waiting for services to start..."
	sleep 15  # Increased wait time to ensure services are fully up
	$(MAKE) setup-minio
	@echo "Local development environment is ready!"
	@echo "Frontend: http://localhost:3000"
	@echo "Lambda API: http://localhost:9090"
	@echo "MinIO Console: http://localhost:9001 (login: minioadmin/minioadmin)"
	@echo "MinIO S3: http://localhost:9000"

clean:
	docker-compose down -v
	docker system prune -f
