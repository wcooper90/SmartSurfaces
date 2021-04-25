import pandas as pd

import xlsxwriter
import sys
import os
sys.path.append(os.path.abspath('../../'))
from src.scrape import scrape_city
from UserInputs import UserInputs


class DF():
    """
    The DF class stores all data on all cities, and has a number of functions
    which allow data to be manipulated and displayed. DF primarily makes calls
    to webscraping functions which fill in data on cities which the City class
    cannot.

    Main functions:
    add_city_values; write_excel; remove_excel; print_df;
    remove_city

    Helper functions:
    return_row

    Note: add_city_values adds web scraped values to the dataframe, and
    add_calculated_city pulls calculated values (such as albedo, greenery)
    in through a City object. remove_city deletes the entire row.

    For any singular run of the code base, only initialization of one DF is necessary.

    """

    # initialize class variables
    def __init__(self, all_columns, scraped_columns, url):
        self.df = pd.DataFrame(columns=all_columns)
        self.columns = all_columns
        self.scraped_columns = scraped_columns
        self.status = 'Empty'
        self.url = url
        self.num_sheets = 0


    # print the first [cols] columns of the dataframe
    def print_df(self, cols=5):
        print(self.df.head(cols))


    # web scrape certain values for a specific city, add to dataframe
    def add_city_values(self, city):

        print(city)

        num_none_values = len(self.columns) - len(self.scraped_columns)
        non_scraped_columns = [x for x in self.columns if x not in self.scraped_columns]

        none_values = {}
        for i in range(len(non_scraped_columns)):
            none_values[non_scraped_columns[i]] = None

        row_values = scrape_city(city, self.scraped_columns, self.url)

        try:
            dic2 = dict(row_values, **none_values)
            self.df = self.df.append(dic2, ignore_index=True)
            print('City of ' + city + ' added to dataframe!')
        except:
            if isinstance(row_values, str):
                print("Error; city is spelled incorrectly or doesn't exist!")
            else:
                print("Error; dataframe rows and the web scraped data are not" +
                        "aligned, or web scraping returned incorrect values!")


    # remove a specified city/row from the dataframe
    def remove_city(self, city):
        return 0


    # write the dataframe to a local excel file
    def write_excel(self):

        try:
            with pd.ExcelWriter(UserInputs.PATH + 'cities' + str(self.num_sheets) + '.xlsx') as writer:
                self.df.to_excel(writer)
            self.num_sheets += 1
            print("Successfully converted to Excel file!")
        except:
            print("Error converting to Excel file!")


    # delete a specified excel file
    def remove_excel(self, sheet_name):
        try:
            os.remove(UserInputs.PATH + sheet_name)
            self.num_sheets -= 1
        except:
            print("Error! Sheet doesn't exist or Python was unable to delete it.")


    # return the row in which a city belongs
    def return_row(self, city_name):

        for i in range(len(self.df)):

            if self.df['City'][i] == city_name:
                return i

        return None
