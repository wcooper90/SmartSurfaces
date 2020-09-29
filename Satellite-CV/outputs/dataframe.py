import pandas as pd
from .scrape import scrape_city
import xlsxwriter
import sys
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


class DF():
    """
    The DF class stores all data on all cities, and has a number of functions
    which allow data to be manipulated and displayed. DF primarily makes calls
    to webscraping functions which fill in data on cities which the City class
    cannot.

    Main functions:
    add_city_values; write_excel; remove_excel; print_df; add_calculated_city;
    remove_city

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

        num_none_values = len(self.columns) - len(self.scraped_columns)
        non_scraped_columns = [x for x in self.columns if x not in self.scraped_columns]

        none_values = {}
        for i in range(len(non_scraped_columns)):
            none_values[non_scraped_columns[i]] = None

        row_values = scrape_city(city, self.scraped_columns, self.url)

        try:
            dic2 = dict(row_values, **none_values)
            self.df = self.df.append(dic2, ignore_index=True)
            print('City of ' + city + ' added!')
        except:
            if isinstance(row_values, str):
                print("Error; city is spelled incorrectly or doesn't exist!")
            else:
                print("Error; dataframe rows and the web scraped data are not" +
                        "aligned, or web scraping returned incorrect values!")


    # web scrape certain values for a specific city, add to dataframe
    def add_calculated_city(self, City):

        return 0


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
