# Billion Dollars Happiness

## Requirements

For development:

- Docker & docker-compose


## How to run

- Create .env files inside the dir
- Use `docker compose up` or `docker compose build` to rebuild images


### Fast restart

You may start in docker only dependencies such as postgres, redis etc.
with command `make docker-run-depends`

> Note: You have to stop previous started containers with `docker compose down`

To run main application use `make run-dev`.
In this case base application is not related to docker and you may use hot reload.
