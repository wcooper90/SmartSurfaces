from bs4 import BeautifulSoup
import sys
import requests
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


def scrape_city(city, columns, url):

    dict_data = {}
    list_data = []

    # initialize data to NoneType
    for i in range(1, len(columns)):
        dict_data[columns[i]] = None

    dict_data[columns[0]] = city
    list_data.append(city)


    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    info = soup.find_all('table')[4].find('tbody')


    city_not_found = True
    for row in info:

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
            if text.rstrip() == city.strip():

                if 'population' in columns:
                    dict_data[columns[1]] = elements[3].get_text().rstrip()
                    list_data.append(dict_data[columns[1]])

                if 'area' in columns:
                    string = ""
                    for char in elements[6].get_text():
                        if not char.isdigit() and char != ".":
                            break
                        try:
                            string += str(char)
                        except:
                            string = str(char)

                    dict_data[columns[2]] = string
                    list_data.append(string)

                if 'location' in columns:

                    coords = elements[10].get_text().split(';')
                    dict_data[columns[3]] = coords[0][-7:]
                    dict_data[columns[3]] += (", " + str(coords[1][2:9]))
                    list_data.append(dict_data[columns[3]])

                city_not_found = False
                break

    if city_not_found:
        return 'error_city_not_found'

    # can return data in list or dictionary form
    return dict_data
