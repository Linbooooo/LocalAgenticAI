.PHONY: test benchmark-agent benchmark-model benchmark-gpu benchmark-cpu swebench doctor install docker-build docker-doctor compose-up compose-gpu compose-cpu compose-pull-model compose-chat

test:
	python3 -m unittest discover -s tests

benchmark-agent:
	python3 scripts/benchmark_agent.py

benchmark-model:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1

benchmark-gpu:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1 --expected-processor gpu --label gpu

benchmark-cpu:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1 --expected-processor cpu --label cpu

swebench:
	python3 scripts/swebench.py --limit 1

doctor:
	python3 -m local_agent doctor

install:
	python3 -m pip install -e .

docker-build:
	docker build -t local-agentic-ai:latest .

docker-doctor:
	docker run --rm -v "$$PWD:/workspace" --network host local-agentic-ai:latest doctor

compose-up: compose-gpu

compose-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate ollama

compose-cpu:
	docker compose -f docker-compose.yml up -d --force-recreate ollama

compose-pull-model:
	docker compose run --rm model-pull

compose-chat:
	docker compose run --rm agent chat
