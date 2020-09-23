import pandas as pd
from scrape import scrape_city

class DF():

    def __init__(self, column_names, url):
        self.df = pd.DataFrame(columns=column_names)
        self.columns = column_names
        self.status = 'Empty'
        self.url = url
        self.num_sheets = 0

    def add_city(self, city):

        row_values = scrape_city(city, self.url)
        self.df.append(row_values, ignore_index=True)


    def remove_city(self, city):
        return 0


    def write_excel(self, new_sheet=True):
        writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

        self.num_sheets += 1
        return 0


    def remove_excel(self, sheet_name):

        self.num_sheets -= 1
        return 0
