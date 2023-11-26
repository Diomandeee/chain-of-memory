# Use the official Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
# This will be the directory where you copy your files to and where CMD will be executed
WORKDIR /usr/src/app

# Copy the contents of the CHAIN_MEMORY directory into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8051 available to the world outside this container
EXPOSE 8051

# Define environment variable (if needed for your application)
ENV NAME World

ENV PYTHONIOENCODING=utf-8

# Run app.py when the container launches
CMD ["python", "./app.py"]
