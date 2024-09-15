FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory in the container
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install GitHub CLI
RUN apt update && apt upgrade && apt-get install -y curl git gh

# Install PyBind
RUN apt install python3-pybind11

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt