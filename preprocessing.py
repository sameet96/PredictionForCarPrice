import os
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

ls = os.chdir('C:\\Users\\samee\\Documents\\carvana')
files = os.listdir(ls)

df = pd.read_csv("data.csv")


print(df.head())

# Check for missing value
msno.matrix(df)
plt.show()

#drop missing values if any
df = df.dropna()

#drop duplicate data
df = df.drop_duplicates()
print(len(df))
# df['year'], df['make'] = df['year'].str.split(" ")[0], df['year'].str.split(" ")[1]
print(df)
#
#Removing Dollar sign from price

df['price'] = df['price'].str[1:]
# print(df['price'].describe())

#Removing ',' from price column
df['price'] = df['price'].str.replace(',', '')
# print(df['price'])
# print(df)
# Extracting Make from year column
# df[['year','make']] = df.year.apply(
#    lambda x: pd.Series(str(x).split(" ")))

space_split = lambda x: pd.Series([i for i in reversed(x.split(' '))])
year = df['year'].apply(space_split)
print (year)
year.rename(columns={0:'make',1:'year'},inplace=True)
print (year)
df['year'] = year['year']
df['make'] = year['make']



df['miles'] = df['miles'].str.replace(' miles', '')
df['miles'] = df['miles'].str.replace(',', '')


df['number_of_keys'] = df['number_of_keys'].astype(int)


#Splitting MPG in highway and city and removing letters from it
mpg_split = lambda x: pd.Series([i for i in reversed(x.split('/'))])
mpg = df['mpg'].apply(mpg_split)
df['highway_mpg'] = mpg[0]
df['city_mpg'] = mpg[1]
df.drop(columns = 'mpg', axis= 1, inplace= True)
df['highway_mpg'] = df['highway_mpg'].str.replace(' Hwy', "")
df['city_mpg'] = df['city_mpg'].str.replace(' City', "")


# Drop terrain
df.drop(columns = 'terrain', inplace= True)

#Getting type of car
df['car_type'] = (df['type_of_car'].str.split(' ').str[-1])


# getting number of doors
df['number_of_doors'] = (df['type_of_car'].str.split(' ').str[0])
df.drop(columns= 'type_of_car', axis=1, inplace= True)

#Getting type of gears
df['type_of_transmission'] = (df['gear_type'].str.split(' ').str[0])
df['type_of_transmission'].value_counts()
df['type_of_transmission'].replace({'Automatic,': 'Auto', 'Auto,': 'Auto', 'Manual,': 'Manual', 'Automatic.': 'Auto', 'Single-Speed': 'Auto'}, inplace= True)

#getting number of gears
import re
def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return " ".join(num)
df['number_of_gears']=df['gear_type'].apply(lambda x: find_number(x))
df['number_of_gears'].value_counts()
df['number_of_gears'].replace({"7 7": "7", "9 9": "9", "8 2": "8", "": "CVT"}, inplace= True)
df.drop(columns= 'gear_type', inplace= True)

#dropping engine type column
df.drop(columns= 'engine_capacity', inplace= True)
len(df)
df.dropna()
len(df)
df

df.to_csv('cleaned_data.csv', index=False)