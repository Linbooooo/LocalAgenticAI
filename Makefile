.PHONY: test doctor install docker-build docker-doctor compose-up compose-pull-model compose-chat

test:
	python3 -m unittest discover -s tests

doctor:
	python3 -m local_agent doctor

install:
	python3 -m pip install -e .

docker-build:
	docker build -t local-agentic-ai:latest .

docker-doctor:
	docker run --rm -v "$$PWD:/workspace" --network host local-agentic-ai:latest doctor

compose-up:
	docker compose up -d ollama

compose-pull-model:
	docker compose run --rm model-pull

compose-chat:
	docker compose run --rm agent chat

