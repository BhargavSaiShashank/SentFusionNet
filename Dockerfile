FROM python:3.14-slim
WORKDIR /app
# Install dependencies from the api folder
COPY improved_version/webapp/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the whole project
COPY . .
# Start the bridge
CMD ["python", "app.py"]
