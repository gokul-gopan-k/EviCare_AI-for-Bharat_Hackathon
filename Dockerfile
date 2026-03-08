# Multi-stage Dockerfile for EviCare Clinical Decision Support System
# Optimized for AWS EC2 t3.micro (1GB RAM, 2 vCPU)

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt . 

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with no cache to minimize layer size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r evicare -g 1000 && \
    useradd -r -u 1000 -g evicare -m -s /bin/bash evicare

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=evicare:evicare backend/ ./backend/
COPY --chown=evicare:evicare frontend/ ./frontend/
COPY --chown=evicare:evicare ingestion/ ./ingestion/
COPY --chown=evicare:evicare vector_db/ ./vector_db/
COPY --chown=evicare:evicare data/ ./data/

# Create directories for volumes with proper permissions
RUN mkdir -p /app/vector_db/chunk_data /app/logs && \
    chown -R evicare:evicare /app

# Copy entrypoint script
COPY --chown=evicare:evicare entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Expose ports
EXPOSE 8000 8501

# Switch to non-root user
USER evicare

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["both"]
