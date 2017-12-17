requirements: ## Install Python Dependencies
	# sudo apt-get install libgdal1-dev python3-gdal
	# export CPLUS_INCLUDE_PATH=/usr/include/gdal
	# export C_INCLUDE_PATH=/usr/include/gdal
	pip3 install --user -r requirements.txt

download-data: ## Download SpaceNet data
	chmod +x ./scripts/download-data.sh
	./scripts/download-data.sh $(city)

clean: ## Clean all generated files
	find . -name "*.pyc" -exec rm {} \;

preprocess: ## Perform preprocessing of images
	python3 ./src/preprocess.py

train: ## Train the model
	python3 ./src/train.py

tensorboard: ## Start TensorBoard
	tensorboard --logdir ./results/summaries

.DEFAULT_GOAL := help
.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
