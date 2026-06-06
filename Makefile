.PHONY: test eval-agentic eval-smoke eval-medium eval-hard benchmark benchmark-gpu benchmark-cpu doctor install docker-build docker-doctor compose-up compose-gpu compose-cpu compose-pull-model compose-chat

test:
	python3 -m unittest discover -s tests

eval-smoke:
	python3 scripts/evaluate_agent.py --suite smoke

eval-agentic:
	python3 scripts/evaluate_agent.py --suite agentic --timeout 300

eval-medium:
	python3 scripts/evaluate_agent.py --suite medium --timeout 300

eval-hard:
	python3 scripts/evaluate_agent.py --suite hard --timeout 300

benchmark:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1

benchmark-gpu:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1 --expected-processor gpu --label gpu

benchmark-cpu:
	python3 scripts/benchmark_ollama.py --runs 5 --warmup 1 --expected-processor cpu --label cpu

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
