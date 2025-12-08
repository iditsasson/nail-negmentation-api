# 1. Base Image: Use a slim Python image for a smaller container
FROM python:3.10-slim

# 2. Set Working Directory: All subsequent commands run inside /app
WORKDIR /app

# 3. Install System Dependencies:
#    - libgl1 is required by OpenCV for headless (no-GUI) processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies:
#    - Copy only the requirements file first to leverage Docker's build cache.
#    - This layer only gets rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Files:
#    - Copy the rest of your app (the .py script and the .pt model file).
#    - Files listed in .dockerignore will be skipped.
COPY . .

# 6. Expose Port:
#    - Tell Docker that the container will listen on port 8000.
EXPOSE 8000

# 7. Run Command:
#    - The command to start the app when the container runs.
#    - We use 4 workers, as recommended in the script.
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]