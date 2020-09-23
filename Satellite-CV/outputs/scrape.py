from bs4 import BeautifulSoup
import sys
import requests
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


def scrape_city(city, columns):

    data = {}

    for i in range(len(columns)):
        data[columns[i]] = None

    page = requests.get(UserInputs.DEFAULT_SCRAPING_URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    info = soup.find_all('table')[4].find('tbody')

    # row_header = True
    # counter = 0
    indecies = {}
    for row in info:

        # if row_header:
        #     row_header = False
        #     continue

        # skip first row
        try:
            elements = row.find_all('td')
        except:
            continue

        for element in elements:
            text = element.get_text()
            # remove wikipedia weird bracketing
            if text[-2] == "]":
                text = text[:-4]

            # rstrip needed because wikipedia has an extra line sometimes
            if element.get_text().rstrip() == city.strip():
                data[columns[0]] = elements[3].get_text()
                data[columns[1]] = elements[6].get_text()
                data[columns[2]] = elements[10].get_text()
                break


    print(data)
    return data

scrape_city("Boston", ['2019 estimate', '2016 land area', 'Location'])
