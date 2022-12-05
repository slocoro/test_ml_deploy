NAME=ml_deploy
USER=$(shell id -u):$(shell id -g)

build:
	docker-compose build $(NAME)

# add -p flag when using docker-compose run because run doesn't map ports from yml file
bash:
	docker-compose run -p 80:80 --rm $(NAME) bash

run:
	docker-compose run -p 80:80 --rm $(NAME)

format: ## Automatically formats code using black and isort
	docker-compose run --rm --user $(USER) linter python3 -m black src/
	docker-compose run --rm --user $(USER) linter python3 -m isort src/