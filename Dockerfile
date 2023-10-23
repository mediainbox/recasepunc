FROM python:3.7.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# install git
RUN apt-get update && apt-get install -y git wget

COPY requirements.txt /app/
COPY requirements-api.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
RUN pip install -r requirements-api.txt

RUN mkdir checkpoint
RUN wget https://github.com/mediainbox/recasepunc/releases/download/v0.1.0/es.24000 -O /app/checkpoint/es.24000

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["python", "main.py"]