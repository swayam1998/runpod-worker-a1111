import os
import time
import requests
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import HfApi
from schemas.input import INPUT_SCHEMA
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.interrogate import INTERROGATE_SCHEMA
from schemas.sync import SYNC_SCHEMA
from schemas.download import DOWNLOAD_SCHEMA
from runpod.serverless.utils import rp_upload
import uuid
import base64

BASE_URI = 'http://127.0.0.1:3000'
TIMEOUT = 600
POST_RETRIES = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()

# Check if the required environment variables are set
required_env_vars = ['BUCKET_ENDPOINT_URL', 'BUCKET_ACCESS_KEY_ID', 'BUCKET_SECRET_ACCESS_KEY']

def are_env_vars_present():
    return all(os.getenv(var) for var in required_env_vars)

# Function to decode base64 image and save it temporarily
def save_base64_image(image_base64, file_name):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        # Save the decoded image to a temporary file
        with open(file_name, 'wb') as f:
            f.write(image_data)
        return file_name
    except Exception as e:
        logger.error(f'Failed to decode and save base64 image: {e}')
        return None

# Upload image if the response contains images and environment variables are set
def upload_images_if_exists(job, response_json):
    image_urls = []
    if are_env_vars_present():
        if 'images' in response_json and isinstance(response_json['images'], list):
            for image_base64 in response_json['images']:
                try:

                    # Generate a unique filename for the image
                    unique_filename = f"{uuid.uuid4()}.png"

                    # Save the base64 image temporarily
                    image_path = save_base64_image(image_base64, unique_filename)
                    
                    if image_path:
                        # Upload the image file to S3
                        image_url = rp_upload.upload_image(job["id"], image_path)
                        image_urls.append(image_url)

                        # Clean up the temporary file
                        os.remove(image_path)
                    else:
                        logger.error(f'Failed to save base64 image for job: {job["id"]}')
                except Exception as e:
                    logger.error(f'Failed to upload image for job: {job["id"]}: {e}')
                    continue
         # Remove 'images' from response_json if images were uploaded successfully
        if image_urls:
            del response_json['images']
    else:
        logger.info('Environment variables for S3 upload are not set, skipping upload.')
    return image_urls

# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload, job_id, retry=0):
    response = session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

    # Retry the post request in case the model has not completed loading yet
    if response.status_code == 404:
        if retry < POST_RETRIES:
            retry += 1
            logger.warn(f'Received HTTP 404 from endpoint: {endpoint}, Retrying: {retry}', job_id)
            time.sleep(0.2)
            send_post_request(endpoint, payload, job_id, retry)

    return response


def validate_input(job):
    return validate(job['input'], INPUT_SCHEMA)


def validate_api(job):
    api = job['input']['api']
    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def validate_payload(job):
    method = job['input']['api']['method']
    endpoint = job['input']['api']['endpoint']
    payload = job['input']['payload']
    validated_input = payload

    if endpoint == 'v1/sync':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, SYNC_SCHEMA)
    elif endpoint == 'v1/download':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, DOWNLOAD_SCHEMA)
    elif endpoint == 'sdapi/v1/txt2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/img2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/interrogate' and method == 'POST':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, INTERROGATE_SCHEMA)

    return endpoint, job['input']['api']['method'], validated_input


def download(job):
    source_url = job['input']['payload']['source_url']
    download_path = job['input']['payload']['download_path']
    process_id = os.getpid()
    temp_path = f"{download_path}.{process_id}"

    # Download the file and save it as a temporary file
    with requests.get(source_url, stream=True) as r:
        r.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Rename the temporary file to the actual file name
    os.rename(temp_path, download_path)
    logger.info(f'{source_url} successfully downloaded to {download_path}', job['id'])

    return {
        'msg': 'Download successful',
        'source_url': source_url,
        'download_path': download_path
    }


def sync(job):
    repo_id = job['input']['payload']['repo_id']
    sync_path = job['input']['payload']['sync_path']
    hf_token = job['input']['payload']['hf_token']

    api = HfApi()

    models = api.list_repo_files(
        repo_id=repo_id,
        token=hf_token
    )

    synced_count = 0
    synced_files = []

    for model in models:
        folder = os.path.dirname(model)
        dest_path = f'{sync_path}/{model}'

        if folder and not os.path.exists(dest_path):
            logger.info(f'Syncing {model} to {dest_path}', job['id'])

            uri = api.hf_hub_download(
                token=hf_token,
                repo_id=repo_id,
                filename=model,
                local_dir=sync_path,
                local_dir_use_symlinks=False
            )

            if uri:
                synced_count += 1
                synced_files.append(dest_path)

    return {
        'synced_count': synced_count,
        'synced_files': synced_files
    }


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(job):
    validated_input = validate_input(job)

    if 'errors' in validated_input:
        return {
            'error': '\n'.join(validated_input['errors'])
        }

    validated_api = validate_api(job)

    if 'errors' in validated_api:
        return {
            'error': '\n'.join(validated_api['errors'])
        }

    endpoint, method, validated_payload = validate_payload(job)

    if 'errors' in validated_payload:
        return {
            'error': '\n'.join(validated_payload['errors'])
        }

    if 'validated_input' in validated_payload:
        payload = validated_payload['validated_input']
    else:
        payload = validated_payload

    try:
        logger.info(f'Sending {method} request to: /{endpoint}', job['id'])

        if endpoint == 'v1/download':
            return download(job)
        elif endpoint == 'v1/sync':
            return sync(job)
        elif method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload, job['id'])

            response_json = response.json()

            # Upload images only if environment variables are present
            image_urls = upload_images_if_exists(job, response_json)

            # Optionally, you can return or log the uploaded image URLs
            if image_urls:
                response_json['s3_image_urls'] = image_urls

        if response.status_code == 200:
            return response_json
        else:
            logger.error(f'HTTP Status code: {response.status_code}', job['id'])
            logger.error(f'Response: {response_json}', job['id'])

            return {
                'error': f'A1111 status code: {response.status_code}',
                'output': response_json,
                'refresh_worker': True
            }
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }


if __name__ == "__main__":
    wait_for_service(f'{BASE_URI}/sdapi/v1/sd-models')
    logger.info('A1111 Stable Diffusion API is ready')
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
