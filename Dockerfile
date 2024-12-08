# Stage 1: Build stage
FROM python:3.12-alpine

# Install build dependencies
RUN apk add --no-cache gcc libffi-dev musl-dev curl bash

# Add project files to install dependencies
WORKDIR /billion
ADD pyproject.toml poetry.lock ./

# Install Poetry and project dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main

# Copy application code and required files
ADD client ./client

# Add entrypoint.sh and grant execution permissions
ADD entrypoint.sh .
RUN chmod +x entrypoint.sh

# Environment variables
ENV WORKERS=1
ENV HOST=0.0.0.0
ENV PORT=80

# Expose the application port
EXPOSE ${PORT}/tcp

# Run the application
CMD ["./entrypoint.sh"]
