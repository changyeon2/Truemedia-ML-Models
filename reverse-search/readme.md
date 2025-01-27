# Reverse Search Project

This repository contains the implementation of the reverse-search pipeline for image deepfake detection.

[Watch an overview](https://www.youtube.com/watch?v=O_n-vwb2chc) of the Machine Learning team discussing this reverse search detector:

[![ML team describes Reverse Search](https://raw.githubusercontent.com/truemediaorg/.github/main/profile/reverse-search-preview.png)](https://www.youtube.com/watch?v=O_n-vwb2chc)

## Project Structure

- **`Dockerfile`**: Defines the Docker image for the application.
- **`classify.py`**: Script for image classification.
- **`docker_run.sh`**: Shell script to run the Docker container.
- **`requirements.txt`**: Python dependencies.
- **`server.py`**: Main sonic server application.

---

## Prerequisites

1. **Docker**: Ensure Docker is installed and running on your system.
2. **API Keys**: Obtain the necessary API keys. You will need 3 API keys: PPLX_API_KEY (for perplexity), OPENAI_API_KEY (for GPT-4), and GOOGLE_APPLICATION_CREDENTIALS (for running reverse image search)

---

## Step-by-Step Guide

### 1. Build the Docker Image

1. Build the Docker image using the `Dockerfile`:
   ```bash
   docker build -t reverse_search_image:<tag> -f Dockerfile .
   ```

### 3. Run the Docker Container

1. Run the docker container with the following command:

   ```bash
   docker run  -d \
            -e PPLX_API_KEY=<your_pplx_api_key> \
            -e OPENAI_API_KEY=<your_openai_api_key> \
            -p 8000:8000 \
            -e SERVER_PORT=8000 \
            -e GOOGLE_APPLICATION_CREDENTIALS="{<your_goole_application_credentials>}" \
            --name "reverse-search" "$DOCKER_REGISTRY/reverse-search:<tag>"
   ```

   Replace `<your_pplx_api_key>`, `<your_openai_api_key>`, and `<your_goole_application_credentials>` with the required API keys.

   The google application credentials should look like the following:

   ```
   '{"type":"service_account","project_id":"...<id>...","private_key_id":"...<id>...","private_key":"...<key>...","client_email":"...<email>...","client_id":"...<id>...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"...<url>...","universe_domain":"googleapis.com"}'
   ```

2. Verify the container is running:
   ```bash
   docker ps
   ```

### 4. Test the Application

1. In a separate terminal, run the following command one or more times

```bash
curl -X GET http://localhost:8000/healthcheck
```

until you see `{"healthy":true}`.

2. Then, test that inference can be run as expected:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     --data '{"file_path":"https://uploads.civai.org/files/jhxTVhsg/b751515306e7.jpg"}'
```

3. Finally, if successful, push the docker image to docker hub:

```bash
docker login

docker push "$DOCKER_REGISTRY/reverse_search_image:<tag>"
```

## Running in GCP

0. Login to your instance. in this case, it is `<PLACEHOLDER_IP>`
1. Pull the Docker Container

```bash
docker pull "$DOCKER_REGISTRY/reverse_search_image:<tag>"
```

For our use case,

```bash
docker pull kevintm1/reverse-search:v1.0
```

1. Run the docker container, as described in [Step 3: Run the Docker Container](#3-run-the-docker-container).

2. Test the model in a separate terminal using:

```bash
curl -X GET http://<PLACEHOLDER_IP>:80/healthcheck
```

and

```bash
curl  -X POST http://<PLACEHOLDER_IP>:80/predict \
      -H "Content-Type: application/json"
      --data '{"file_path":"https://uploads.civai.org/files/jhxTVhsg/b751515306e7.jpg"}'
```

The expected output should be in the format

```bash
{"score":0,"rationale":"The image appears ......","sourceUrl":"..."}
```
