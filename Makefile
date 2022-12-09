APP ?= notion-semantic-search
PROJECT ?= cohere-sandbox-360501
COMPONENT ?= backend
VERSION ?= stable
REGISTRY ?= gcr.io/$(PROJECT)
IMAGE ?= $(REGISTRY)/$(APP)-$(COMPONENT)


download-bbc-news:
	poetry run python scripts/download_xlsum.py

clean:
	rm -r data
	rm embeddings.npz

embeddings:
	poetry run embed csv --root-dir data --output-file embeddings.npz

build-data: download-bbc-news embeddings

build-server:
	@DOCKER_DEFAULT_PLATFORM='linux/amd64' docker build --build-arg port=8080 -t $(IMAGE):$(VERSION) .

build-server-data: embeddings.npz
	-@docker rm $(APP)-$(COMPONENT)-data 2>/dev/null
	@docker create --name $(APP)-$(COMPONENT)-data $(IMAGE):$(VERSION)
	@docker cp embeddings.npz $(APP)-$(COMPONENT)-data:/data/embeddings.npz
	@docker commit $(APP)-$(COMPONENT)-data $(IMAGE):$(VERSION)

build: build-server build-server-data

shell: build
	@DOCKER_DEFAULT_PLATFORM='linux/amd64' docker run -e COHERE_TOKEN=$(COHERE_TOKEN) -it $(IMAGE):$(VERSION) /bin/bash

run: build
	@DOCKER_DEFAULT_PLATFORM='linux/amd64' docker run -e COHERE_TOKEN=$(COHERE_TOKEN) -p 8080:8080 -it $(IMAGE):$(VERSION)
