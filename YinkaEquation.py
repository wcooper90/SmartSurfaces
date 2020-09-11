import numpy as np
import requests
import bs4


# scrape the necessary data for Yinka's equation
def scrape():
    return 0


# implementation of Yinka's equation
def compute_CO2(delta_albedo, area, average_energy_over_area, natural_albedo = 0.3):
    albedo_inversion = 1 - delta_albedo
    unit_conversion1 = (average_energy_over_area * 1000) / 24
    unit_conversion2 = unit_conversion1 * albedo_inversion * delta_albedo / 5.35

    CO2_concentraion = 413 * np.exp(unit_conversion2)
    unit_conversion3 = CO2_concentraion * area/(510.1 * np.power(10, 12)) * (7.77 * np.power(10, 12))

    return unit_conversion3


if __name__ == "__main__":
    kilos_CO2 = compute_CO2(0.05, 1000, 3)
    print("CO2 saved: " + str(kilos_CO2))
