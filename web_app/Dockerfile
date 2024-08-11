# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the flask_app folder into the container at /app
COPY . /app

# Install any necessary dependencies specified in requirements.txt within flask_app
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Flask app runs on
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
