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

import base64
from supabase import create_client
from urllib.parse import urlparse, parse_qs

BASE_URI = 'http://127.0.0.1:3000'
TIMEOUT = 600
POST_RETRIES = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()


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

def create_supabase_client():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)
    return supabase

def upload_to_supabase(image_data_base64, userId, genId, fileName):
    supabase = create_supabase_client()

    # Decode the base64 string into bytes
    image_data = base64.b64decode(image_data_base64)

    # Define the path and bucket
    bucket_name = "proprofiles"
    file_path = f'v1/generations/generatedImages/{userId}/{genId}/{fileName}'
    print(file_path)

    # Upload the image synchronously
    upload_response = supabase.storage.from_(bucket_name).upload(file_path, image_data, {"content-type": "image/png"})
    logger.info(f"upload_response: {upload_response.json()}")

    # Assuming upload_response is an object with a 'status_code' and a method to get json like 'json()'
    if upload_response.status_code == 200:
        response_data = upload_response.json()  # Get the JSON content of the response
        if 'error' in response_data:
            logger.error(f"Failed to upload image: {response_data['error']['message']}")
            return None
        else:
            # Construct the URL of the uploaded image
            public_url = file_path
            return public_url
    else:
        logger.error("Failed to upload image due to HTTP error.")
        return None



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

        if response.status_code == 200:
            webhook = job["webhook"]
            if webhook:
                #"https://example.com/webhooks/inference?genId=12345&userId=67890"

                # Parse the URL to get the query parameters
                parsed_url = urlparse(webhook)
                query_params = parse_qs(parsed_url.query)

                # Extract the 'genId' and 'userId' from the query parameters
                genId = query_params.get('genId', [None])[0]
                userId = query_params.get('userId', [None])[0]

                if genId and userId:
                    # Get the generated images from response json
                    data = response.json()
                    images = data['images']
                    upload_urls = []

                    for index, image_base64 in enumerate(images):
                        # Assuming the image is in base64 format
                        fileName = f"{index}.png"
                        logger.info(fileName)

                        # Upload each image to Supabase storage
                        upload_url = upload_to_supabase(image_base64, userId, genId, fileName)
                        upload_urls.append(upload_url)
                    
                    # Replace the 'images' in the response with 'upload_urls'
                    logger.info(upload_urls)
                    data['images'] = upload_urls
        
                    # Return the modified response
                    return {
                        'error': None,
                        'output': data,
                        'status': 'complete'
                    }

                        
            return response.json()
        else:
            logger.error(f'HTTP Status code: {response.status_code}', job['id'])
            logger.error(f'Response: {response.json()}', job['id'])

            return {
                'error': f'A1111 status code: {response.status_code}',
                'output': response.json(),
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
