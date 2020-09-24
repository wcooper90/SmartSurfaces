import pandas as pd
from .scrape import scrape_city
import xlsxwriter
import sys
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


class DF():

    def __init__(self, all_columns, scraped_columns, url):
        self.df = pd.DataFrame(columns=all_columns)
        self.columns = all_columns
        self.scraped_columns = scraped_columns
        self.status = 'Empty'
        self.url = url
        # self.url = UserInputs.DEFAULT_SCRAPING_URL
        self.num_sheets = 0


    def print_df(self, cols=5):
        print(self.df.head(cols))


    def add_city(self, city):

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


    def remove_city(self, city):
        return 0


    def write_excel(self, new_sheet=True):

        try:
            with pd.ExcelWriter(UserInputs.PATH + 'cities.xlsx') as writer:
                self.df.to_excel(writer, sheet_name='Sheet' + str(self.num_sheets))
            self.num_sheets += 1
            print("Successfully converted to Excel file!")
        except:
            print("Error converting to Excel file!")


    def remove_excel(self, sheet_name):

        self.num_sheets -= 1
        return 0
