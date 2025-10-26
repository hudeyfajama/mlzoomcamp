FROM agrigorev/zoomcamp-model:2025

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy the FastAPI application
COPY app.py ./

# Expose the port
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
