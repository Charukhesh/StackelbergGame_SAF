def emissions_aviation_fuel():
    return 9.5 #kg CO2 per gallon

def emissions_saf():
    return 3.78 #kg CO2 per gallon

#usual subsidies - $1.25/gallon
#usual tax - $90/ton CO2

def revenue():
    return 9.7 #billion dollars

def operating_cost():
    return 5.66 #billion dollars without fuel

def fuel_qty():
    return 1.495 #billion gallons

def aviation_fuel_price(qty): # https://farmdocdaily.illinois.edu/wp-content/uploads/2024/12/fdd122624.pdf
    p = 2.10 #2.10*(qty/1.5)**(-10) # qty in billions of gallons p in $/gallon
    return p

def saf_price(qty):
    p = 6.3 #6.3*(qty/0.0045)**(-8) # qty in billions of gallons p in $/gallon
    return p
