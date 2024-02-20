import csv
import os
import requests
import json
from bs4 import BeautifulSoup

# Function to search for images using BeautifulSoup
def search_images(query):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_tags = soup.find_all('img')
    image_urls = [img['src'] for img in image_tags]
    return image_urls[1]  # return the URL of the first image (ignoring the thumbnail)

# Function to download and save image
def download_image(url, folder, filename):
    response = requests.get(url)
    with open(os.path.join(folder, filename), 'wb') as f:
        f.write(response.content)

car_brands = ['audi', 'bmw', 'ford', 'hyundi', 'merc', 'skoda', 'toyota', 'vauxhall', 'vw']

# Initialize folder numbering
folder_number = 1
folder_mapping = {}

for car_brand in car_brands:

    #csv_file = 'test.csv'
    csv_file = f"Ident/{car_brand}.csv"
    output_folder = 'data'
    # Read the CSV file
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a dictionary to store folder numbers for unique combinations of column 1 and column 2
   
    # Counter for file naming
    file_number = 1
    
    # Flag to skip the header row
    is_header = True
    
    # Open the CSV file for reading and writing
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if exists
        
        for row in reader:
            search_query = f"{car_brand} {row[0]} {row[1]} {row[8]}L {row[5]}"  # Construct the search query
            image_url = search_images(search_query)  # Get URL of the first image
            filename = f"{car_brand}_{file_number}.jpg"  # Construct filename with file number
            
            # Determine the folder path based on the unique combination of column 1 and column 2
            folder_key = f"{car_brand}{row[0]}"
            if folder_key not in folder_mapping:
                folder_mapping[folder_key] = len(folder_mapping) + 1
                folder_path = os.path.join(output_folder, f"{len(folder_mapping)}")
                os.makedirs(folder_path)
            else:
                folder_path = os.path.join(output_folder, str(folder_mapping[folder_key]))
            
            # Download and save the image
            download_image(image_url, folder_path, filename)
            print(search_query  + " Saved to " + folder_path)
            
            file_number += 1  # Increment file number
    
# File path to save the dictionary
output_json = 'folder_mapping.json'

# Writing dictionary to JSON file
with open(output_json, 'w') as jsonfile:
    json.dump(folder_mapping, jsonfile)
    
# File path to save the CSV file
output_csv = 'folder_mapping.csv'

# Writing dictionary to CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=folder_mapping.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data
    writer.writerow(folder_mapping)
        
print("Images downloaded and sorted successfully. Folder mapping files created.")
