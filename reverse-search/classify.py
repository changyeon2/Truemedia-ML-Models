import os
from dotenv import load_dotenv
import requests
from openai import OpenAI
from urllib.parse import urlparse
import base64
from PIL import Image
import io
import time
import json
from pydantic import BaseModel, Field
from google.cloud import vision
from google.oauth2 import service_account
from google.cloud.vision_v1 import types


# Load the environment variables
load_dotenv()

def set_gcloud():
    try:
        credential_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        # Parse the credentials from the environment variable
        credentials_info = json.loads(credential_json)
        try:
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        except Exception as e:
            raise ValueError("Failed to create credentials from the provided JSON info!") from e

        # Initialize the ImageAnnotatorClient with the provided credentials
        try:
            client = vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as e:
            raise RuntimeError("Failed to initialize ImageAnnotatorClient!") from e

        return client

    except Exception as e:
        print(f"Error in setting up Google Cloud client: {e}")
        return None


def download_file(input_path):
    """
    Download a file from a given URL and save it locally in an 'images' folder.
    If input_path is a local file path and the file exists, skip the download.
    :param input_path: The URL of the file to download or a local file path.
    :return: The local filepath to the downloaded or existing file.
    """
    # Create 'images' folder if it doesn't exist
    images_folder = 'images'
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created '{images_folder}' folder.")

    if input_path.startswith(('http://', 'https://')):
        filename = input_path.split('/')[-1].split('?')[0]
        local_filepath = os.path.join(images_folder, filename)
        if os.path.isfile(local_filepath):
            print(f"The file already exists locally: {local_filepath}")
            return local_filepath
        start_time = time.time()
        response = requests.get(input_path, stream=True)
        response.raise_for_status()
        with open(local_filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        end_time = time.time()
        print(f"Downloaded file saved to {local_filepath} in {end_time - start_time:.2f} seconds.")
    else:
        # If it's a local path, ensure it's in the images folder
        filename = os.path.basename(input_path)
        local_filepath = os.path.join(images_folder, filename)
        if input_path != local_filepath:
            # If the file is not already in the images folder, copy it there
            if os.path.isfile(input_path):
                import shutil
                shutil.copy2(input_path, local_filepath)
                print(f"Copied file to {local_filepath}")
            else:
                raise FileNotFoundError(f"No such file: '{input_path}'")
        elif not os.path.isfile(local_filepath):
            raise FileNotFoundError(f"No such file: '{local_filepath}'")
        print(f"Using existing file: {local_filepath}")
    return local_filepath

# Function to check if the URL is a social media link
def is_exclude(url, exclude_list):
    for domain in exclude_list:
        if domain in url:
            return True
    return False

def get_search_results(image_url, exclude_list=None, engine='google'):
    client = set_gcloud()
    image = types.Image()
    image.source.image_uri = image_url

    # Perform web detection request
    response = client.web_detection(image=image)
    web_detection = response.web_detection

    # Check if there are pages with matching images
    if web_detection.pages_with_matching_images:
        for page in web_detection.pages_with_matching_images:
            # Check if the page URL is not a social media link
            if exclude_list:
                if not is_exclude(page.url, exclude_list):
                    return page.url
            else:
                return page.url
    print("No results found or all results were filtered out for ", image_url)
    return None


def get_pplx_text(link, engine='perplexity'):
    api_key = os.getenv('PPLX_API_KEY')
    if not api_key:
        print("perplexity api key not found")

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    messages = [
        {"role": "system", "content": "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with a user."},
        {"role": "user", "content": f"Please give me the detailed text content on this page: {link}. Make sure only to use this page as reference and do not make anything up that is not in this page."}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred while querying Perplexity API: {str(e)}")
        return None


def encode_image(local_image_path):
    try:
        with Image.open(local_image_path) as image:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {local_image_path}: {e}")
        return None
    

class ValidResponse(BaseModel):
    explanation: str = Field(..., description="Detailed explanation of your reasoning.")
    classification: int = Field(..., description="Return 0 for real or 1 for fake. If for some reason you cannot classify it, return -1")
    # maybe we don't want to give it the option of -1 here


def get_GPT_prediction(local_image_path, page_link, engine='gpt'):
    if page_link:
        response_text = get_pplx_text(page_link)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("openai api key not found")

    client = OpenAI(api_key=api_key)

    try:
        base64_image = encode_image(local_image_path)
        if base64_image is None:
            return -1, "Error in encoding image."
        if page_link:
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Evaluate whether this image is real or fake based on the image and the following context: \n {response_text} \n Take into consideration only the image and the context provided to you, and do not make up any information or use any information that is not in the context provided."
                }
            ]
        else:
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Was this image generated by an AI or otherwise digitally manipulated?"
                }
            ]

                    
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an advanced image forgery detection model."},
                {"role": "user", "content": content}
            ],
            response_format=ValidResponse,
        )
        gpt_response = response.choices[0].message
        if gpt_response.parsed:
            print("gpt_response.parsed", gpt_response.parsed)
            gpt_response = gpt_response.parsed
        elif gpt_response.refusal:
            print("response refusal")
            print(gpt_response.refusal)
            return -1, "Error, model refusal."

        return gpt_response.classification, gpt_response.explanation
    except Exception as e:
        print(f"Error in API call: {e}")
        return -1, f"Error in API call: {e}"


def classify_image(inputs):
    image_url = inputs.get('file_path', None)
    if not image_url:
        return {"error": "incorrect filepath"}
    # Download the image
    local_image_path = download_file(image_url)
    print(local_image_path)

    # Get search results (placeholder for now)
    exclude_list = ['x.com', 'twitter.com', 'youtube.com', 'facebook.com', 'instagram.com', 'reddit.com', 'medium.com']
    page_link = get_search_results(image_url, exclude_list=exclude_list)
    # page_link = "https://www.theweek.in/news/biz-tech/2024/07/15/thomas-matthew-crooks-blackrock-inc-takes-down-ad-featuring-trump-shooter.html"
        
    try:
        prediction, explanation = get_GPT_prediction(local_image_path, page_link)
        
        # Check for error cases
        if prediction == -1:
            return {"error": explanation}  # Use the explanation as the error message

        # Prepare results dictionary
        results = {"score": prediction, "rationale": explanation, "sourceUrl": page_link}
        return results
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    test_input = {'file_path': "https://nytco-assets.nytimes.com/2021/08/Joe-hero.jpg"}
    results = classify_image(test_input)
    print(results)