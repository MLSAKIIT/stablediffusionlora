import requests
from requests import exceptions
import os

# Extract images from Bing Search Using API
def get_images(query, api_key, Path, Max_Images=50, Group_Size=50):
    URL = "https://api.bing.microsoft.com/v7.0/images/search"
    EXCEPTIONS = {
        IOError,
        FileNotFoundError,
        exceptions.RequestException,
        exceptions.HTTPError,
        exceptions.ConnectionError,
        exceptions.Timeout,
    }
    headers = {"Ocp-Apim-Subscription-Key" : api_key}
    params = {"q": query, "count": Group_Size, "offset": 0}
    response = requests.get(URL, headers=headers, params=params) # Send request to Bing API
    response.raise_for_status() # Check for any errors

    results = response.json() # Get the JSON response
    total = max(0, min(Max_Images, results["totalEstimatedMatches"])) # Get the total number of images
    print(f"Found {total} results for {query}")

    # Create a directory to store the images

    if not os.path.exists(f"{Path}\Images"):
        os.makedirs(f"{Path}\Images")
    if not os.path.exists(f"{Path}\ImageCaptions"):
        os.makedirs(f"{Path}\ImageCaptions")

    IMAGE_DIRECTORY = f"{Path}\Images"
    IMAGE_CAPTION_DIRECTORY = f"{Path}\ImageCaptions"

    # Download the images
    counter = 0
    for i in results["value"]:
        counter += 1
        try:
            # Get the image
            # print(f"Downloading {i}")
            # Download image with tqdm
            response = requests.get(i["contentUrl"], timeout=30, stream=True)
            response.raise_for_status() # Check for error
            ext = i["encodingFormat"] # Get the extension of the image
            file_name = counter # Get the image ID
            print(f"Downloaded {file_name}.{ext}")
            f = open(f"{IMAGE_DIRECTORY}\{file_name}.{ext}", "wb") # Open the file
            f.write(response.content) # Write the image to the file
            f.close() # Close the file

            # Image Caption file
            f = open(f"{IMAGE_CAPTION_DIRECTORY}\{file_name}.txt", "a") # Open the file
            f.write(query) # Write the caption to the file
            f.close() # Close the file

        except Exception as e:
            if type(e) in EXCEPTIONS:
                print(f"Skipping: {e}")
                continue

def main():
    Query = "Pikachu" # Query to search
    API_KEY = os.environ.get("API_KEY") # Bing API Key
    Path = "CONTRIBUTIONS\Sanjeev-Kumar78\src\Dataset\Dataset" # Path to store the images
    get_images(Query, API_KEY, Path) # Get the images

if __name__ == "__main__":
    main()
