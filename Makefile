NAME=ml_deploy
USER=$(shell id -u):$(shell id -g)

build:
	docker-compose build $(NAME)

# add -p flag when using docker-compose run because run doesn't map ports from yml file
bash:
	docker-compose run --rm $(NAME) bash

run-server:
	docker-compose run -p 80:80 --rm $(NAME)

format: ## Automatically formats code using black and isort, doesn't work right now because of permission issues on some files
	#docker-compose run --rm --user $(USER) linter python3 -m black src/
	#docker-compose run --rm --user $(USER) linter python3 -m isort src/
	docker-compose run --rm linter python3 -m black src/
	docker-compose run --rm linter python3 -m isort src/

install:
	python3 -m pip install -r requirements.txt

pytest:
	docker-compose run --rm $(NAME) pytest tests/