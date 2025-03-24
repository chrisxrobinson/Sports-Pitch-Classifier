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

show-localstack-logs:
	docker-compose logs localstack

restart-localstack:
	docker-compose restart localstack
	sleep 10
	$(MAKE) setup-localstack

setup-localstack:
	chmod +x ./scripts/setup_localstack.sh
	./scripts/setup_localstack.sh || true

upload-models:
	@echo "Uploading models to S3..."
	@if [ -z "$$S3_BUCKET" ]; then \
		echo "Error: S3_BUCKET environment variable not set."; \
		echo "Usage: S3_BUCKET=your-bucket-name make upload-models"; \
		exit 1; \
	fi; \
	if [ ! -d "./backend/models" ]; then \
		echo "No models directory found at ./backend/models/"; \
		exit 1; \
	fi; \
	for model in ./backend/models/*.pth; do \
		if [ -f "$$model" ]; then \
			echo "Uploading $$(basename $$model) to s3://$$S3_BUCKET/model/"; \
			aws s3 cp "$$model" "s3://$$S3_BUCKET/model/$$(basename $$model)"; \
		fi \
	done; \
	echo "Upload complete!"

up:
	docker-compose up --build -d
	@echo "Waiting for services to start..."
	sleep 5
	$(MAKE) setup-localstack
	@echo "Local development environment is ready!"
	@echo "Frontend: http://localhost:3000"
	@echo "Lambda API: http://localhost:9090"
	@echo "LocalStack (AWS Services): http://localhost:4566"
	@echo ""
	@echo "TROUBLESHOOTING COMMANDS:"
	@echo "- View Lambda logs:     make show-lambda-logs"
	@echo "- View Frontend logs:   make show-frontend-logs"
	@echo "- View LocalStack logs: make show-localstack-logs"
	@echo "- View all logs:        make show-all-logs"
	@echo ""
	@echo "If you experience issues, try: make down && make local-dev"

down:
	docker-compose down -v
	docker system prune -f

show-lambda-logs:
	docker-compose logs lambda

show-frontend-logs:
	docker-compose logs frontend

show-all-logs:
	docker-compose logs