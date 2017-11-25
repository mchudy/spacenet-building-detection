requirements: ## Install Python Dependencies
	pip3 install --user -r requirements.txt

download-data: ## Download SpaceNet data
	chmod +x ./scripts/download-data.sh
	./scripts/download-data.sh $(city)

clean: ## Clean all generated files
	find . -name "*.pyc" -exec rm {} \;

.DEFAULT_GOAL := help
.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'