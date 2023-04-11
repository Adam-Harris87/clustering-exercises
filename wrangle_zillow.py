import numpy as np
import pandas as pd
import os
import env

def acquire_zillow():
    '''
    This function will retrieve zillow home data for 2017 properties. It will only get
    single family residential properties. the function will attempt to open the data from 
    a local csv file, if one is not found, it will download the data from the codeup
    database. An env file is needed in the local directory in order to run this file.
    '''
    if os.path.exists('zillow_2017.csv'):
        print('opening data from local file')
        df = pd.read_csv('zillow_2017.csv', index_col=0)
    else:
        # run sql query and write to csv
        print('local file not found')
        print('retrieving data from sql server')
        query = '''
SELECT *
FROM properties_2017
JOIN predictions_2017
	USING (parcelid)
LEFT JOIN airconditioningtype
	USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype
	USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype
	USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype
	USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype
	USING (propertylandusetypeid)
LEFT JOIN storytype
	USING (storytypeid)
LEFT JOIN typeconstructiontype
	USING (typeconstructiontypeid)
WHERE latitude IS NOT NULL
	AND longitude IS NOT NULL
    AND parcelid IN (
		SELECT parcelid FROM unique_properties)
;
        '''
        connection = env.get_db_url('zillow')
        df = pd.read_sql(query, connection)
        df.to_csv('zillow_2017.csv')
    
    # renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'garagecarcnt':'cars_garage',
                              'garagetotalsqft':'garage_sqft',
                              'lotsizesquarefeet':'lot_size',
                              'poolcnt':'pools',
                              'regionidcity':'region',
                              'yearbuilt':'year_built',
                              'taxvaluedollarcnt':'tax_value'
                              })
    return df

def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

def get_single_unit(df):
    homes = ((df.propertylandusedesc =='Single Family Residential') |
            (df.propertylandusedesc == 'Mobile Home') |
            (df.propertylandusedesc =='Manufactured, Modular, Prefabricated Homes'))
    df = df[homes]
    return df

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''
    This function takes in a dataframe, the percent of columns and rows
    that need to have values/non-nulls
    and returns the dataframe with the desired amount of nulls left.
    '''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df