# Set image name as a variable
FROM ultralytics/ultralytics:latest

# set a directory for the app
WORKDIR /app

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
