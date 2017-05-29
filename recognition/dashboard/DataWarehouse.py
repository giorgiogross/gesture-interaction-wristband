
# ================================================================
# Application: DataWarehouse - Storing and manipulating input data
# ================================================================


# Load packages for econometric analysis

import numpy as np
import datetime as dt
import pandas as pd
from datetime import datetime, timedelta
import math as mth
import pickle
from Databases import Database
from pandas.tseries.offsets import CustomBusinessDay
#import holidays
from datetime import datetime, timedelta
import matplotlib
import math
import matplotlib.pyplot as plt
# Preferences

matplotlib.style.use('ggplot')
pd.set_option('display.expand_frame_repr', False)




def time_offset_rolling_sum_df_ser(data_df_ser, window_i_s, min_periods_i=1, center_b=False):
    """ Function that computes a rolling sumness

    Credit goes to user2689410 at http://stackoverflow.com/questions/15771472/pandas-rolling-mean-by-time-interval

    # follows logic of time_offset_rolling_std_df_ser

    """

    def calculate_sum_at_ts(ts):
        if center_b == False:
            dslice_df_ser = data_df_ser[
                ts-pd.datetools.to_offset(window_i_s).delta+timedelta(0,0,1):
                ts
            ]
            # adding a microsecond because when slicing with labels start and endpoint
            # are inclusive
        else:
            dslice_df_ser = data_df_ser[
                ts-pd.datetools.to_offset(window_i_s).delta/2+timedelta(0,0,1):
                ts+pd.datetools.to_offset(window_i_s).delta/2
            ]
        if  (isinstance(dslice_df_ser, pd.DataFrame) and dslice_df_ser.shape[0] < min_periods_i) or \
            (isinstance(dslice_df_ser, pd.Series) and dslice_df_ser.size < min_periods_i):
            return dslice_df_ser.sum()*np.nan   # keeps number format and whether Series or DataFrame
        else:
            return dslice_df_ser.sum()


    if isinstance(window_i_s, int):
        sum_df_ser = pd.rolling_sum(data_df_ser, window=window_i_s, min_periods=min_periods_i, center=center_b)
    elif isinstance(window_i_s, str):
        idx_ser = pd.Series(data_df_ser.index.to_pydatetime(), index=data_df_ser.index)
        sum_df_ser = idx_ser.apply(calculate_sum_at_ts)

    return sum_df_ser




def time_offset_rolling_std_df_ser(data_df_ser, window_i_s, min_periods_i = 1, center_b = False):

    """
    - Goal:    calculate rolling std. - Credit goes to user2689410 at http://stackoverflow.com/questions/15771472/pandas-rolling-mean-by-time-interval

    :param data_df_ser:     If a DataFrame is passed, the time_offset_rolling_std_df_ser is computed for all columns.
    :param window_i_s:      int or string.  - If int is passed, window_i_s is the number of observations used for calculating
                                            the statistic, as defined by the function pd.time_offset_rolling_std_df_ser()
                                            - If a string is passed, it must be a frequency string, e.g. '90S'. This is
                                            internally converted into a DateOffset object, representing the window_i_s size.
    :param min_periods_i:   Minimum number of observations in window_i_s required to have a value.
    :param center_b:        - False:    output-timestamp - last value of the given window
                            - True:     output-timestamp - centered value of the given window
    :return: data_df_ser (pd.Dataframe): with calculated volatilities.
    """

    def calculate_std_at_ts( ts ):

        """Function (closure) to apply that actually computes the rolling std"""

        if center_b == False:


            # Slice data frame: E.g. last 5 min
            #
            # For ts = 09:05:00 (2010-01-04)
            #
            # 2010-01-04 09:00:30         NaN
            # 2010-01-04 09:01:00    0.217163
            # 2010-01-04 09:01:30    0.145072
            # 2010-01-04 09:02:00   -0.010714
            # 2010-01-04 09:02:30    0.007366
            # 2010-01-04 09:03:00    0.093702
            # 2010-01-04 09:03:30   -0.074285
            # 2010-01-04 09:04:00   -0.002008
            # 2010-01-04 09:04:30    0.009038
            # 2010-01-04 09:05:00   -0.043858
            # Name: ln_return, dtype: float64
            #

            # numpy std:
            #
            # In[13]:             np.std([[ 0.217163, 0.145072, -0.010714, 0.007366, 0.093702, -0.074285, -0.002008, 0.009038, -0.043858]])
            # Out[13]: 0.089272406274002819

            # numpy mean:
            #
            # In[21]: np.mean([[ 0.217163, 0.145072, -0.010714, 0.007366, 0.093702, -0.074285, -0.002008, 0.009038, -0.043858]])
            # Out[21]: 0.037941777777777769 (NOT zero!)
            #

            # pandas std:
            #
            # In[10]: test = pd.DataFrame([0.217163, 0.145072, -0.010714, 0.007366, 0.093702, -0.074285, -0.002008, 0.009038, -0.043858])
            # In[11]: test.std()
            # Out[11]: 0    0.094688 x
            #

            # Moench, p. 13:  'Five-minute moving sums of squared tick-by-tick returns'
            #
            # In[19]: sum([0.217163**2, 0.145072**2, (-0.010714)**2, 0.007366**2, 0.093702**2, (-0.074285)**2, (-0.002008)**2, 0.009038**2, (-0.043858)**2])
            # Out[19]: 0.08468226920600001 = Vol
            #
            # Alternative calculation:
            # In[10]: test = pd.DataFrame([0.217163, 0.145072, -0.010714, 0.007366, 0.093702, -0.074285, -0.002008, 0.009038, -0.043858])
            # (test**2).sum() = 0    0.084682
            #
            # Note, (marginal) smaller than moving sums calculation
            #

            dslice_df_ser = data_df_ser[
                ts - pd.datetools.to_offset(window_i_s).delta + timedelta(0,0,1):
                ts
            ]
            # adding a microsecond because when slicing with labels start and endpoint
            # are inclusive

        else:
            dslice_df_ser = data_df_ser[
                ts - pd.datetools.to_offset(window_i_s).delta / 2 + timedelta(0,0,1):
                ts + pd.datetools.to_offset(window_i_s).delta / 2
            ]

        # If data type is a data frame and length lower then min_periods then return nan
        if  ( isinstance(dslice_df_ser, pd.DataFrame) and dslice_df_ser.shape[0] < min_periods_i ) or \
            ( isinstance(dslice_df_ser, pd.Series) and dslice_df_ser.size < min_periods_i ):

            return dslice_df_ser.std() * np.nan   # keeps number format and whether Series or DataFrame

        else:

            # DataFrame.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs)
            # Return sample standard deviation over requested axis.
            #
            # Normalized by N-1 by default. This can be changed using the ddof argument
            #

            #return dslice_df_ser.std() # Frederik Impl
            return (dslice_df_ser**2).sum() # Elmar Impl

    # End of method


    if isinstance(window_i_s, int):  # Use pandas rolling std function (if 'int')
        std_df_ser = pd.rolling_std(data_df_ser, window = window_i_s, min_periods = min_periods_i, center = center_b)


    elif isinstance(window_i_s, str): # Use own function (if 'str')

        # 2010-01-04 09:00:30
        # 2010-01-04 09:01:00
        # 2010-01-04 09:01:30
        #

        idx_ser = pd.Series(data_df_ser.index.to_pydatetime(), index = data_df_ser.index)


        # 2010-01-04 09:00:30         NaN
        # 2010-01-04 09:01:00         NaN
        # 2010-01-04 09:01:30    0.050976
        # 2010-01-04 09:02:00    0.116472
        # 2010-01-04 09:02:30    0.109810
        # 2010-01-04 09:03:00    0.095115
        # 2010-01-04 09:03:30    0.108463
        # 2010-01-04 09:04:00    0.102020
        # 2010-01-04 09:04:30    0.095767
        # 2010-01-04 09:05:00    0.094688 x
        # 2010-01-04 09:05:30    0.089857
        #


        std_df_ser = idx_ser.apply(calculate_std_at_ts)

    return std_df_ser




# =======================================================
# Application: DataWarehouse - for Data Manipulation
# =======================================================



class DataWarehouse:

    def __init__(self, dataSource, dataSelector, asset_class):
        """
        - Goal:             init DataWarehouse object
        -  DataWarehouse:   location where data is stored and input parameters for Event_Study_Toolbox are calculated.
        -  DataWarehouse:   can also be loaded with pre-calculated input parameters

        :param dataSource:      string. "local_raw", "local_load", "database"
        :param dataSelector:    string. If dataSource == "database", this should be the query to select the options data. If dataSource == "local_raw" or "local_load" , this should be the address of the file (respective directory).
        :param asset_class:     string. "Equity", "Future"
        """


        self.asset_class  = asset_class
        self.tick = "not defined"
        self.open_time = "not defined"
        self.close_time = "not defined"
        self.range = "not defined"
        self.h = "not defined"
        self.min = "not defined"
        self.sec = "not defined"


        if dataSource == 'local_raw':
            self.data = pd.read_csv(dataSelector)
            self.data['Date'] = pd.to_datetime(self.data['loctimestamp'], format='%Y-%m-%d %H:%M:%S')
            self.data['time'] = self.data['Date']
            self.data = self.data.set_index('Date')
            print('Data loaded')

        if dataSource == 'local_load':
            filePath = dataSelector
            with open(filePath, "rb") as file:
                self.data = pickle.load(file)

        if dataSource == 'database':
            print('via Database')
            db = Database("risqdata")
            value = np.asarray(db.getValuesFromTable(dataSelector))
            self.data = pd.DataFrame(value[0:, 1], columns=['price'], index=value[0:, 0], dtype=float)


            # self.data:
            # raw price data, 15 sec
            #
            #                        price
            # 2000-06-29 09:00:15  5229.42
            # 2000-06-29 09:00:30  5227.98
            # 2000-06-29 09:00:45  5227.97
            # 2000-06-29 09:01:00  5228.86



        print("Datawarehouse set up with " + asset_class)



    def preparation(self, open_time, close_time, tick): # set the opening hours for consistent datastream

        """
        - GOAL:         set opening hours and tick size of the index.
        - preparation:  all other timestamps are dropped -> leads to consistent timeseries for each day.
        - preparation:  resamples data due to Ticksize.
        - preparation:  impicitly calls def tickSize() - to store ticksize (self.h, self.min, self.min) and self.range (dt.time() of openinghours in hours and min) in WareHouse object

        :param open_time:   dt.time().  Timestamp of first tick of each day.
        :param close_time:  dt.time().  Timestamp of last tick of each day.
        :param tick:        sting.      Ticksize.
        :return:            no return. stores all parameters in DataWarehouse object
        """

        self.tick = tick

        self.open_time = open_time

        self.close_time = close_time

        #                      price
        # 2000-06-29 09:00:15  5229.42
        # 2000-06-29 09:00:30  5227.98
        # 2000-06-29 09:00:45  5227.97


        # Now, group price data for every DAY
        #
        # [                      price
        # 2010-01-04 09:00:15  2974.86
        # 2010-01-04 09:00:30  2976.09 x
        # 2010-01-04 09:00:45  2979.67
        # 2010-01-04 09:01:00  2982.56 x
        # 2010-01-04 09:01:15  2985.73
        # 2010-01-04 09:01:30  2986.89 x

        DFList = [group[1] for group in self.data.groupby(self.data.index.date)]


        # Next, resample the data, e.g. 30 sec
        #
        # [                      price
        # 2010-01-04 09:00:30  2976.09
        # 2010-01-04 09:01:00  2982.56
        # 2010-01-04 09:01:30  2986.89
        #

        b = [x.resample(tick, how = 'last', closed = 'right', label = 'right') for x in DFList]

        self.data = pd.concat(b)


        # Select data between user specified 'open' and 'close' time
        #
        # 2015-06-30 17:29:30  3446.35
        # 2015-06-30 17:30:00  3440.33
        #
        # [1428931 rows x 1 columns]
        #

        self.data = self.data.ix[self.data.index.indexer_between_time(open_time, close_time)]


        # Translate tick size string to int and calculate the (intra-day) range of the time series
        #

        self.tickSize()

        print("Preparation done! With trading hours from " + open_time.strftime('%H:%M:%S') + " to " + close_time.strftime('%H:%M:%S') + " and ticksize " + tick + ".")





    def tickSize(self):

        """
        - GOAL:     to store ticksize (self.h, self.min, self.min) and self.range (timespan of openinghours in hours and min) in WareHouse object
        :return:    no return. stores all parameters in DataWarehouse object
        """

        # Calculate range

        t_a = dt.datetime(1990, 1, 1, self.open_time.hour, self.open_time.minute, self.open_time.second)

        t_b = dt.datetime(1990, 1, 1, self.close_time.hour, self.close_time.minute, self.close_time.second)

        t = t_b - t_a # e.g. 8:30:00

        # Hours
        hp = t.seconds / (3600) # 8.5

        self.range = dt.time(math.floor(hp), int(hp % 1 * 60)) # 08:30:00

        # Tick size, e.g. 15 sec
        help = self.data.index[1] - self.data.index[0] # 00:00:30

        divisor = help.seconds # 30 (sec)

        self.h = int(divisor/3600) # 0

        self.min = int(divisor/60 - self.h*60) # 0

        self.sec = int(divisor - self.min*60 - self.h*3600) # 30 (Translate string '30S' to int)





    def calculateReturns(self):
        """
        - GOAL: calculate log-returs.
        :return: no return. stores all parameters in DataWarehouse object
        """


        #                        price  ln_return
        # 2010-01-04 09:00:30  2976.09        NaN
        # 2010-01-04 09:01:00  2982.56   0.217163
        # 2010-01-04 09:01:30  2986.89   0.145072
        #

        # Check:
        #
        # In[8]: np.log(2982.56/2976.09)
        # Out[8]: 0.0021716336897469844
        #

        self.data['ln_return'] = (np.log(self.data['price']) - np.log(self.data['price'].shift(1))) * 100

        print('Log_Returns calculated')




    def calculateCumRe(self):

        """
        - GOAL: calculate cumulative returns in course of each day.
        :return: no return. stores all parameters in DataWarehouse object
        """

        # Group price data for every DAY (like above)
        #

        DFList = [group[1] for group in self.data.groupby(self.data.index.date)]

        # For every day...
        for df in DFList:

            # ... calculate cumulative sum (of log-returns)
            df['cum_re'] = df['ln_return'].cumsum()



        # [                       price  ln_return    cum_re
        # 2010-01-04 09:00:30  2976.09        NaN       NaN
        # 2010-01-04 09:01:00  2982.56   0.217163  0.217163
        # 2010-01-04 09:01:30  2986.89   0.145072  0.362235
        # 2010-01-04 09:02:00  2986.57  -0.010714  0.351521
        #

        self.data = pd.concat(DFList)



    def calculateVol(self, window, min_period = 1):

        """
        - GOAL:             calculate volatility on a rolling window basis in course of the specified window.
        :param window:      string. rolling window size.
        :param min_period:  int.    minimum of observations to calculate volatility. If no value is given -> default min_period = 1
        :return:            no return. stores all parameters in DataWarehouse object
        """

        #self.data['squared'] = self.data['ln_return']**2

        # Group price data for every DAY (like above)
        #

        DFList = [group[1] for group in self.data.groupby(self.data.index.date)]

        # For every day...
        for df in DFList:

            #  ... calculate volatility (std) (of log-returns)
            df['vol'] = time_offset_rolling_std_df_ser( df['ln_return'], window_i_s = window, min_periods_i = min_period )



        self.data = pd.concat(DFList)




    def setInvestigationHorizont(self, start, end):

        """
        - Goal: specify investigation horizont
        :param start:
        :param end:
        :return:
        """

        start = self.data.index.searchsorted(start) # Translate start date to position

        end = self.data.index.searchsorted(end) # Translate end date to position


        #                        price  ln_return
        # 2010-01-04 09:00:30  2976.09        NaN
        # 2010-01-04 09:01:00  2982.56   0.217163
        # ...
        # 2015-06-29 17:29:30  3472.43   0.047817
        # 2015-06-29 17:30:00  3471.91  -0.014976
        #
        # Note, last day (30.06.2015) NOT included)

        self.data = self.data.ix[start:end]

        print("Data for investigation horizont is provided!")




    def showPrice_Return(self, plt_type):

        """
        -Goal: Plots price, log-return or both of the index.
        :param plt_type: string.    -"price"            plot price
                                    -"ln_return"        plot log-returns
                                    -"ln_return_price"  sub-plot price and log-returns
        :return: no return.
        """


        if (plt_type == 'price'):

            self.data['price'].plot(subplots = True, style = 'b', figsize = (8,7))
            plt.title("Prices of the Index" )
            plt.ylabel('price')


        elif (plt_type == 'ln_return'):

            self.data['ln_return'].plot(subplots = True, style = 'b', figsize = (8,7))
            plt.title("Log-Return" )
            plt.ylabel('return in %')


        elif (plt_type == 'ln_return_price'):

            self.data[['price', 'ln_return']].plot(subplots = True, style = 'b', figsize = (8,7))





    def courseOfAday(self, plot, type):

        """
        - Goal: calculate and plot average day for cumulative log-return, volatility
        :param plot: boolean.   -True:  plot average day and return result in a pandas Dataframe
                                -False: just return result in a pandas Dataframe

        :param type: string. "vol", "cum_re"
        :return: pandas Dataframe
        """

        if self.close_time == "not defined" or self.open_time == "not defined" or self.tick == "not defined":

            print("Please set close_time, open_time and tick via preparation or manually! ")

        else:

            # Group price data by time (NOT date)
            #

            # [                       price  ln_return    cum_re       vol
            # 2010-01-21 09:00:00  2914.68  -0.028815 -0.028815  0.000830
            # 2010-03-09 09:00:00  2879.14   0.023274  0.023274  0.000542
            # 2010-03-26 09:00:00  2948.15   0.027818  0.027818  0.000774

            DFList2 = [group[1] for group in self.data.groupby(self.data.index.time)]


            # pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)
            # Return a fixed frequency datetime index, with day (calendar) as the default frequency
            #

            # DatetimeIndex(['2016-09-13 09:00:00', '2016-09-13 09:00:30',
            #               '2016-09-13 09:01:00', '2016-09-13 09:01:30',...

            dates = pd.date_range(start = self.open_time.strftime('%H:%M:%S'), end = self.close_time.strftime('%H:%M:%S'), freq = self.tick)

            plt_cum = pd.DataFrame()
            plt_cum['time'] = dates
            plt_cum = plt_cum.set_index('time')

            plt_cum['mean'] = 0.0
            plt_cum['std'] = 0.0
            plt_cum['#Obs'] = 0.0


            #                       mean  std  #Obs
            # time
            # 2016-09-13 09:00:00   0.0  0.0   0.0
            # 2016-09-13 09:00:30   0.0  0.0   0.0
            # 2016-09-13 09:01:00   0.0  0.0   0.0
            #

            plt_cum


            # For every time...
            for df in DFList2:

                t3 = df.index[0].time() # 09:00:00

                plt_cum['mean'][t3] = df[type].mean()
                plt_cum['std'][t3] = df[type].std()
                plt_cum['#Obs'][t3] = df[type].__len__() - np.isnan(df[type]).sum()



            plt_cum['CI_minus'] = plt_cum['mean'] - plt_cum['std'] * 1.959964 / np.sqrt(plt_cum['#Obs'])
            plt_cum['CI_plus'] = plt_cum['mean'] + plt_cum['std'] * 1.959964 / np.sqrt(plt_cum['#Obs'])

            if plot:

                plt_cum['mean'].plot(subplots = True, style = 'b', figsize = (8, 7))
                plt_cum['CI_minus'].plot(subplots = True, style = 'c', figsize = (8, 7))
                plt_cum['CI_plus'].plot(subplots = True, style = 'c', figsize = (8, 7))

                plt.fill_between(x = plt_cum.index, y1 = plt_cum['CI_plus'].values, y2 = plt_cum['CI_minus'].values, )

                plt.axis('tight')
                plt.title(type + " in course of one day")

                if type == 'cum_re':
                    plt.ylabel("Returns in %")

                elif type == 'vol':
                    plt.ylabel("Volatility")

                else:
                    plt.close()
                    print("Please specify right type: cum_re or vol")


            return plt_cum[['mean', 'CI_plus', 'CI_minus']]