.PHONY: run run-gpu run-cpu once once-gpu once-cpu

COMPOSE := docker compose
COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

# Prefer GPU when Docker can schedule a GPU container; otherwise same stack on CPU.
run:
	@docker run --rm --gpus all busybox true >/dev/null 2>&1 \
		&& { echo "Using GPU (docker-compose.gpu.yml)"; $(COMPOSE_GPU) run --rm allinone; } \
		|| { echo "Using CPU (GPU not available to Docker)"; $(COMPOSE) run --rm allinone; }

run-gpu:
	$(COMPOSE_GPU) run --rm allinone

run-cpu:
	$(COMPOSE) run --rm allinone

once:
	@docker run --rm --gpus all busybox true >/dev/null 2>&1 \
		&& $(COMPOSE_GPU) run --rm allinone --pretty -p "$(PROMPT)" \
		|| $(COMPOSE) run --rm allinone --pretty -p "$(PROMPT)"

once-gpu:
	$(COMPOSE_GPU) run --rm allinone --pretty -p "$(PROMPT)"

once-cpu:
	$(COMPOSE) run --rm allinone --pretty -p "$(PROMPT)"
