# Use an official Python runtime as a parent image
FROM python:3.11.5-slim

# Set the working directory to /app
WORKDIR /app
#RUN apt-get update && apt-get install -y libpq-dev build-essential

# Copy the current directory contents into the container at /app
COPY . /app
RUN apt-get update && apt-get install -y ca-certificates  && apt-get install -y curl

RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "unstructured[docx]"

#RUN curl https://ollama.ai/install.sh | sh


ARG OLLAMA_MODEL_NAME="mistral:instruct"
ENV OLLAMA_MODEL_NAME=$OLLAMA_MODEL_NAME

# Expose the port that Gradio will run on
EXPOSE 7860


#RUN ollama pull $OLLAMA_MODEL_NAME 

CMD ["python", "app.py"]