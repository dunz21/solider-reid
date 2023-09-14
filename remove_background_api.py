import os
import requests

API_KEY_REMOVE_BG="P662RFB22N3KjvsVJVvzsDWf"
API_KEY_REMOVAL_AI="65025df79d72b1.97958148"

def remove_background(base_dir='images_copy', api_key='YOUR_API_KEY'):
    paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(base_dir) for f in filenames if f.endswith('.png')]
    paths = sorted(paths)

    output_dir = base_dir + '_background'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in paths:
        relative_path = os.path.relpath(img_path, base_dir)  # Get relative path to preserve subdirectory structure in the output directory
        output_img_path = os.path.join(output_dir, relative_path)

        output_folder = os.path.dirname(output_img_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(img_path, 'rb') as file:
            response = removebg(file)
            if response.status_code == 200:
                with open(output_img_path, 'wb') as out_file:
                    out_file.write(response.content)
            else:
                print(f"Error processing {img_path}. Status code: {response.status_code}")

def removebg(file):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': file},
        data={'size': 'auto'},
        headers={'X-Api-Key': API_KEY_REMOVE_BG},
    )
    return response

def removalai(file):
    response = requests.post(
        "https://api.removal.ai/3.0/remove",
        headers={"Rm-Token": API_KEY_REMOVAL_AI},
        files={"image_file": file},
        data={"get_file": "1"}
    )
    return response


remove_background(api_key="65025df79d72b1.97958148")
