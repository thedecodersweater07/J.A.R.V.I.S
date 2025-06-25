# JARVIS Docker Setup

This directory contains the Docker configuration for the JARVIS AI system.

## Prerequisites

- Docker Engine 20.10.0+
- Docker Compose 1.29.0+
- (Optional) NVIDIA Container Toolkit for GPU acceleration

## Services

The following services are defined in the Docker Compose configuration:

1. **api**: Main JARVIS API service (development mode with hot-reload)
   - Port: 8000

2. **ml-service**: Machine Learning service for model inference
   - Mounts: `./ml` and `./data`

3. **postgres**: PostgreSQL database
   - Port: 5432
   - User: jarvis
   - Password: jarvis
   - Database: jarvis

4. **redis**: Redis for caching and message brokering
   - Port: 6379

5. **pgadmin**: Web-based PostgreSQL administration (optional)
   - Port: 5050
   - Email: admin@jarvis.local
   - Password: admin

6. **redis-commander**: Web-based Redis administration (optional)
   - Port: 8081

## Getting Started

1. **Build and start all services**:
   ```bash
   docker-compose -f docker/docker-compose.yml up --build
   ```

2. **Start services in detached mode**:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

3. **View logs**:
   ```bash
   # All services
   docker-compose -f docker/docker-compose.yml logs -f
   
   # Specific service
   docker-compose -f docker/docker-compose.yml logs -f api
   ```

4. **Stop services**:
   ```bash
   docker-compose -f docker/docker-compose.yml down
   ```

## Development Workflow

- The `api` service mounts your local code directory, so changes are reflected immediately
- Use `docker-compose -f docker/docker-compose.yml restart api` to restart just the API service
- The development environment includes hot-reload for Python code

## Environment Variables

- `JARVIS_ENV`: Set to `development` or `production`
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `ML_MODELS_DIR`: Directory for ML models (default: `/home/appuser/app/data/models`)

## Volumes

- `postgres_data`: Persistent storage for PostgreSQL
- `redis_data`: Persistent storage for Redis
- `ml_models`: Storage for ML models (mounted to `./data/models`)

## Troubleshooting

1. **Port conflicts**:
   - Check if ports 8000, 5432, 6379, 5050, or 8081 are already in use

2. **Build issues**:
   - Run `docker system prune -af` to clean up unused containers and images
   - Increase Docker's allocated resources in Docker Desktop settings

3. **Database connection issues**:
   - Wait for PostgreSQL to fully initialize (healthcheck may take a minute)
   - Check logs: `docker-compose -f docker/docker-compose.yml logs postgres`

## Production Deployment

For production, you should:

1. Set `JARVIS_ENV=production`
2. Use proper secrets management (e.g., Docker secrets or environment files)
3. Configure proper TLS/SSL termination
4. Set up proper monitoring and logging
5. Consider using a production-grade orchestrator like Kubernetes
