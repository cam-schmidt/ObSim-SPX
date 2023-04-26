import pandas as pd
import random
import numpy as np
from scipy import interpolate
from scipy.stats import norm

# Simulation Parameters
n = 100000   # Simulations

# Load the CSV file into a pandas DataFrame
call_df = pd.read_csv(r'/Users/camschmidt/Desktop/SPX_C.csv')
put_df = pd.read_csv(r'/Users/camschmidt/Desktop/SPX_P.csv')  


# Select rows with root value equals to 'SPX'
call_df_spx = call_df[call_df['root'] == 'SPX']
put_df_spx = put_df[put_df['root'] == 'SPX']

# Remove the 'Greeks' columns 
call_df_spx = call_df_spx.drop(['delta', 'gamma', 'theta', 'vega', 'rho'], axis=1)         
put_df_spx = put_df_spx.drop(['delta', 'gamma', 'theta', 'vega', 'rho'], axis=1)  

# Rename columns
call_df_spx = call_df_spx.rename(columns={'option price': 'call price', 'implied volatility': 'call implied volatility'})
put_df_spx = put_df_spx.rename(columns={'option price': 'put price', 'implied volatility': 'put implied volatility'})

# merge the call and put dataframes into one dataframe
options_df = pd.merge(call_df_spx, put_df_spx, on=['quote_date', 'root', 'exdate', 'strike price', 'underlying price', 'time to maturity'])


def clean_data(df):
    # Condition 1: remove rows where 'time to maturity' < 7 or > 365
    condition1 = (df['time to maturity'] >= 7) & (df['time to maturity'] <= 365)
    df = df[condition1]
    
    return df

options_df = clean_data(options_df)
pd.set_option('display.max_rows', None)

# Sort the dataframe by quote_date, root, time to maturity, and strike price
options_df = options_df.sort_values(by=['quote_date', 'root', 'time to maturity', 'strike price'])
options_df = options_df.reset_index(drop=True)

Date =  #input date
Maturity = #input maturity

def options_dict(root, quote_date, time_to_maturity):
    # filter the dataframe for the given root, quote_date, and time_to_maturity
    filtered = options_df[(options_df['root'] == root) & (options_df['quote_date'] == quote_date) & (options_df['time to maturity'] == time_to_maturity)]
    
    # create the dictionary
    result = {}
    for index, row in filtered.iterrows():
        key = (row['call price'], row['put price'])
        value = row['strike price']
        result[key] = value
        
    return result
  

  
  class MonteCarlo:    
    def __init__(self, root, quote_date, time_to_maturity):
        global r, T

        self.root = root
        self.quote_date = quote_date
        self.time_to_maturity = time_to_maturity 
        self.options_dictionary = self.generate_dictionary()
        T = time_to_maturity / 365
        r = r_values[int(quote_date.split('-')[0])]  # select the appropriate rate given the year
        
    def generate_dictionary(self):
        # filter the dataframe for the given root, quote_date, and time_to_maturity
        filtered = options_df[(options_df['root'] == self.root) & (options_df['quote_date'] == self.quote_date) & (options_df['time to maturity'] == self.time_to_maturity)]
    
        # create the dictionary
        result = {}
        for index, row in filtered.iterrows():
            key = (row['call price'], row['put price'])
            value = row['strike price']
            result[key] = value
        
        return result
    
    def loss_minimizer(self, a):
        min_pair = None
        min_val = float('inf')
        
        for pair in self.options_dictionary.keys():
            call_price, put_price = pair
            loss = (a * np.exp(r*T) * call_price) + ((1 - a) * np.exp(r*T) * put_price)
            
            if loss < min_val:
                min_val = loss
                min_pair = pair
    
        
        return round(self.options_dictionary[min_pair], 2)
    
    def run_simulations(self):
        return [self.loss_minimizer(random.uniform(0, 1)) for i in range(n)]
    
simulated_stock_data = MonteCarlo('SPX', Date, Maturity).run_simulations()


class ObSim:
    def __init__(self, K):
        self.K = K
        
    def call(self):
        payoff_list = [max(ST - self.K, 0) for ST in simulated_stock_data]
        obsim_price = np.exp(-r*T) * (sum(payoff_list) / len(simulated_stock_data))
        obsim_price = round(obsim_price, 4)
        
        return obsim_price
        
    def put(self):
        payoff_list = [max(self.K - ST, 0) for ST in simulated_stock_data]
        obsim_price = np.exp(-r*T) * (sum(payoff_list) / len(simulated_stock_data))
        obsim_price = round(obsim_price, 4)
        
        return obsim_price

        
    def confidence_interval(self, type, c_level):
        if type == 'c':
            mew = self.call()
        elif type == 'p':
            mew = self.put()
        
        
        st_dev = np.std(simulated_stock_data)
        z_score = norm.ppf(c_level + (1 - c_level) / 2)
        ci_lower_bound = round(mew - z_score * (st_dev/np.sqrt(len(simulated_stock_data))), 4)
        ci_upper_bound = round(mew + z_score * (st_dev/np.sqrt(len(simulated_stock_data))), 4)
        
        return (ci_lower_bound, ci_upper_bound)
      
      
search_date = '2016-08-29'
underlying_price = options_df.loc[options_df['quote_date'] == search_date, 'underlying price'].iloc[0]


sample_strikes = [1900, 1950, 2000, 2125, 2150, 2175,
                 2200, 2250, 2300, 2350, 2400]

# Create a column of real data call prices for the sample strikes 
real_call_values = result = [key[0] for key, value in dictionary_test.items() if value in sample_strikes]

# Creating Obsim call prices for the sample strikes 
obsim_sample_values = [ObSim(K).call() for K in sample_strikes]

# Creating Obsim confidence intervals for the sample strikes 
obsim_sample_ci = [ObSim(K).confidence_interval('c', .95) for K in sample_strikes]


df_1 = pd.DataFrame(sample_strikes, columns=['Strike, K'])
df_1['Underlying Price'] = underlying_price
df_1['Time to Maturity'] = (f'{round(T * 365, 2)} days')
df_1['Real Call Price'] = real_call_values
df_1['ObSim Price'] = obsim_sample_values
df_1['95% C.I.'] = obsim_sample_ci
df_1['Relative % Error'] = round(((df_1['ObSim Price'] - df_1['Real Call Price']) / ((df_1['ObSim Price'] + df_1['Real Call Price'])/2)) * 100, 2)

df_1
