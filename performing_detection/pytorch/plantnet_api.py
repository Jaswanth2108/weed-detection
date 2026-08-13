import requests
import io
from PIL import Image

PLANTNET_API_KEY = "2b10MFLFcPzB67x0WDdgDOCEO"
PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/all"

def identify_plant_species(image_input, organ="leaf", api_key=PLANTNET_API_KEY):
    """
    Sends an image to Pl@ntNet API for plant species identification.
    
    :param image_input: File-like object (UploadedFile / BytesIO), PIL Image, or file path.
    :param organ: Organ type ('leaf', 'flower', 'fruit', 'bark', 'auto').
    :param api_key: Pl@ntNet API Key.
    :return: dict with formatted results or error message.
    """
    try:
        # Prepare image bytes
        if isinstance(image_input, (str, io.BytesIO)) or hasattr(image_input, 'read'):
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            if hasattr(image_input, 'getvalue'):
                img_bytes = image_input.getvalue()
            elif isinstance(image_input, str):
                with open(image_input, 'rb') as f:
                    img_bytes = f.read()
            else:
                img_bytes = image_input.read()
        elif isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.save(buf, format='JPEG')
            img_bytes = buf.getvalue()
        else:
            return {"success": False, "error": "Unsupported image format"}

        params = {"api-key": api_key}
        files = [('images', ('plant.jpg', img_bytes, 'image/jpeg'))]
        data = {'organs': [organ]}

        response = requests.post(PLANTNET_API_URL, params=params, files=files, data=data, timeout=15)
        
        if response.status_code == 200:
            res_json = response.json()
            results = res_json.get("results", [])
            
            parsed_results = []
            for item in results[:5]: # Top 5 predictions
                score = round(item.get("score", 0) * 100, 2)
                species = item.get("species", {})
                scientific_name = species.get("scientificNameWithoutAuthor", "Unknown")
                authorship = species.get("scientificNameAuthorship", "")
                common_names = species.get("commonNames", [])
                family = species.get("family", {}).get("scientificNameWithoutAuthor", "")
                genus = species.get("genus", {}).get("scientificNameWithoutAuthor", "")
                
                # Extract image URLs
                example_images = []
                for img in item.get("images", []):
                    url_obj = img.get("url", {})
                    if "m" in url_obj:
                        example_images.append(url_obj["m"])
                    elif "o" in url_obj:
                        example_images.append(url_obj["o"])

                parsed_results.append({
                    "score": score,
                    "scientific_name": scientific_name,
                    "authorship": authorship,
                    "common_names": common_names,
                    "family": family,
                    "genus": genus,
                    "images": example_images[:3]
                })

            return {
                "success": True,
                "best_match": res_json.get("bestMatch", "Unknown"),
                "results": parsed_results
            }
        elif response.status_code == 401:
            return {"success": False, "error": "Invalid Pl@ntNet API Key (401 Unauthorized)"}
        elif response.status_code == 404:
            return {"success": False, "error": "No plant match found by Pl@ntNet API"}
        else:
            return {"success": False, "error": f"API Error ({response.status_code}): {response.text}"}
            
    except Exception as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
