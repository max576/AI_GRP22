# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:47:55 2024

@author: max
"""

import requests
import csv

def query_dvla_api(registration_number, api_key):
    url = 'https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles'
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'registrationNumber': registration_number
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

def main():
    api_key = 'wJRB6Riu0kOJZNAs11rb8K651hFfuvU9IxqsJaf8'
    registration_number = 'KO21NTD'
    data = query_dvla_api(registration_number, api_key)
    print(data)
    write_to_csv(data, 'number_plate_data.csv')

if __name__ == '__main__':
    main()
