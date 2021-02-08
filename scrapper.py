# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:43:27 2020

@author: samee
"""

import csv
# import requests
import pandas as pd
from bs4 import BeautifulSoup
import math
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException
import numpy as np
import threading
start_time = time.time()
driver = webdriver.Chrome('D:\Projects\chromedriver.exe')
# driver.get('https://www.carvana.com/cars?page=1')
# TotalCars = driver.find_element_by_xpath(
#     "//span[@class = 'paginationstyles__PaginationText-mpry3x-5 klmwCt']").text
# getTotalCars = TotalCars.split('of ')[1].split(" ")[0]
# print(getTotalCars)
# gettingPages = (int(getTotalCars) / 21)
# frac, whole = math.modf(gettingPages)
# pagesToScan = whole - 10
# print(pagesToScan)
#
# pageNumber = 1
# finalVehicleId = []
# while pageNumber < pagesToScan:
#     newUrl = 'https://www.carvana.com/cars?page={}'.format(pageNumber)
#     driver.get(newUrl)
#     soup = BeautifulSoup(driver.page_source, "html.parser")
#     for i in soup.find_all('div', {
#         'class': 'ShowroomResultTile__ShowroomTileWrapper-n5q2qt-0 fobSgr styles__ResultTileWrapper-sc-1algal3-0 gUbYju'}):
#         link = i.find('a', href=True)
#         if link is None:
#             continue
#         print(link)
#         vehicleId = link['href']
#
#         vehicleId = vehicleId.split('/')[2]
#         print(vehicleId)
#         finalVehicleId.append(vehicleId)
#     pageNumber += 1
#
# print(finalVehicleId)
# df = pd.DataFrame(finalVehicleId)
# df.to_csv('vehicleId.csv', sep=',', header=True)


def getText(xpath):
    textValue = driver.find_element_by_xpath(xpath)
    if bool(textValue.text):
        finalVal = textValue.text
    else:
        finalVal = None
    return finalVal

def checkForElement(xpath1):
    check = len(driver.find_elements_by_xpath(xpath1)) > 0
    return check
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


def getData(attributeName):
    global textValue
    genericToGetData = "//*[contains(text(), 'property')]//following-sibling::p"
    if check_exists_by_xpath(genericToGetData.replace("property", "%s" % attributeName)):
        if bool(driver.find_element_by_xpath(genericToGetData.replace("property", "%s" % attributeName)).text):
            textValue = driver.find_element_by_xpath(genericToGetData.replace("property", "%s" % attributeName)).text
        else:
            textValue = None
    else:
        textValue = None
    return textValue




import csv

# with open('vehicleIdStored.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)
#
# print(data)


carsDone = 1
# finalVehicleId= pd.read_csv('vehicleIdStored.csv', sep = ',')
# test_list = finalVehicleId.values.tolist()
#
# test_list = [int(i) for i in test_list]
# for i in range(0, len(test_list)):
#     test_list[i] = int(test_list[i])
# final = str(test_list)
# print(finalVehicleId)

# carsToIterate = 106
# totalChunk = 1
def getDatFromDf(finalVehicleId, fileName):
    modelName = []
    typeOfVehicle = []
    engineType = []
    mpg = []
    year_n_make = []
    exteriorColor = []
    interiorColor = []
    transmission = []
    doors = []
    numberOfKeys = []
    price = []
    miles = []
    vehicle = 0
    chunks = 10
    print("before deleting duplicates " + str(len(finalVehicleId)))
    finalVehicleId = finalVehicleId.drop_duplicates()
    print("after deleting duplicates " + str(len(finalVehicleId)))
    for (index, rows) in finalVehicleId.iterrows():
        # print(x[1:-1])
        # print(y)
        # x = x.astype(str)
        # x = x.lstrip()
        # x= x.rstrip()
        # print(x)
        rows[0]
        carUrl = 'https://www.carvana.com/vehicle/{}'.format(rows[0])
        print(carUrl)
        driver.get(carUrl)
        if (checkForElement("//*[contains(text(), 'Engine Type')]//following-sibling::p") and checkForElement("//*[contains(text(), 'MPG')]//following-sibling::p") and checkForElement("//*[contains(text(), 'Exterior Color')]//following-sibling::p") and checkForElement("//*[contains(text(), 'Interior Color')]//following-sibling::p") and
                checkForElement("//*[contains(text(), 'Transmission')]//following-sibling::p") and checkForElement("//*[contains(text(), 'Number of Keys')]//following-sibling::p") and checkForElement("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[1]") and
                checkForElement("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[2]") and checkForElement("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[3]") and checkForElement(
                    "//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Price-v7qvvn-12 iwnRgB jctOAy biSZho']") and (checkForElement(
                    "//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Mileage-v7qvvn-14 iwnRgB jctOAy fxPHFg']") or checkForElement("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[3]"))):
            driver.get(carUrl)
            engineType.append(getData("Engine Type"))
            mpg.append(getData("MPG"))
            exteriorColor.append(getData("Exterior Color"))
            interiorColor.append(getData("Interior Color"))
            transmission.append(getData("Transmission"))
            doors.append(getData("Doors"))
            numberOfKeys.append(getData("Number of Keys"))
            year_n_make.append(getText("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[1]"))
            modelName.append(getText("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[2]"))

            price.append(getText(
                "//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Price-v7qvvn-12 iwnRgB "
                "jctOAy biSZho']"))
            print(bool(len(driver.find_elements_by_xpath("//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Mileage-v7qvvn-14 iwnRgB "
                "jctOAy fxPHFg']"))))
            print(checkForElement("//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Mileage-v7qvvn-14 iwnRgB "
                "jctOAy fxPHFg']"))
            if checkForElement("//span[@class = 'styles__Wrapper-nkchis-0 hszKUY']"):
                miles.append(getText("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[3]").split("|")[1])
                typeOfVehicle.append(getText("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[3]").split("|")[0])
            elif checkForElement("//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Mileage-v7qvvn-14 iwnRgB "
                "jctOAy fxPHFg']"):
                typeOfVehicle.append(getText("(//div[@class = 'styles__MakeModelAndTrim-v7qvvn-8 YwKtb']//div)[3]"))
                miles.append(getText(
                    "//div[@class = 'styles__TextLabel-v7qvvn-1 styles__TextLabelRight-v7qvvn-2 styles__Mileage-v7qvvn-14 iwnRgB "
                    "jctOAy fxPHFg']"))
            vehicle += 1

            print("Vehicles scrapped" + str(vehicle) + "from chunk number" + str(chunks))
    # soup = BeautifulSoup(driver.page_source, "html.parser")
    # div = soup.find('div', {'data-qa': 'vehicle-details-and-price'})
    # children = div.findChildren("div" , recursive=False)
    # for child in children:
    #   print(child)
    #     carsDone += 1
    #     print(carsDone)
    print(year_n_make, modelName, typeOfVehicle, engineType, mpg, exteriorColor, interiorColor, transmission, doors,
        numberOfKeys, price, miles)
    infoDict = {"year and Make": year_n_make, "Model": modelName, "vehicle Type": typeOfVehicle, "Engine": engineType,
                "MPG": mpg,
                "Exterior Color": exteriorColor, "Interior Color": interiorColor, "Transmission": transmission,
                "Doors": doors, "Keys": numberOfKeys, "price": price, "Miles": miles}

    df = pd.DataFrame(infoDict)
    df.to_csv(fileName, sep=',', header=False, mode= 'a')
    chunks += 1

    # totalChunk += 1
    # print(totalChunk)
finalVehicleId= pd.read_csv('vehicleIdStored.csv', sep = ',')
df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21 = np.array_split(finalVehicleId, 21)
def Main():
	# threads = [] # Threads list needed when we use a bulk of threads
    # mythread1 = threading.Thread(target=getDatFromDf(df1, "Data1.csv"))
    # mythread2 = threading.Thread(target=getDatFromDf(df2, "Data2.csv"))
    # mythread3 = threading.Thread(target=getDatFromDf(df3, "Data3.csv"))
    # mythread4 = threading.Thread(target=getDatFromDf(df4, "Data4.csv"))
    # mythread5 = threading.Thread(target=getDatFromDf(df5, "Data5.csv"))
    # mythread6 = threading.Thread(target=getDatFromDf(df6, "Data6.csv"))
    # mythread7 = threading.Thread(target=getDatFromDf(df7, "Data7.csv"))
    # mythread8 = threading.Thread(target=getDatFromDf(df8, "Data8.csv"))
    # mythread9 = threading.Thread(target=getDatFromDf(df9, "Data9.csv"))
    # mythread10 = threading.Thread(target=getDatFromDf(df10, "Data10.csv"))
    mythread11 = threading.Thread(target=getDatFromDf(df11, "Data11.csv"))
    mythread12 = threading.Thread(target=getDatFromDf(df12, "Data12.csv"))
    mythread13 = threading.Thread(target=getDatFromDf(df13, "Data13.csv"))
    mythread14 = threading.Thread(target=getDatFromDf(df14, "Data14.csv"))
    mythread15 = threading.Thread(target=getDatFromDf(df15, "Data15.csv"))
    mythread16 = threading.Thread(target=getDatFromDf(df16, "Data16.csv"))
    mythread17 = threading.Thread(target=getDatFromDf(df17, "Data17.csv"))
    mythread18 = threading.Thread(target=getDatFromDf(df18, "Data18.csv"))
    mythread19 = threading.Thread(target=getDatFromDf(df19, "Data19.csv"))
    mythread20 = threading.Thread(target=getDatFromDf(df20, "Data20.csv"))
    mythread21 = threading.Thread(target=getDatFromDf(df21, "Data21.csv"))



    # mythread1.start()
    # mythread2.start()
    # mythread3.start()
    # mythread4.start()
    # mythread5.start()
    # mythread6.start()
    # mythread7.start()
    # mythread8.start()
    # mythread9.start()
    # mythread10.start()
    mythread11.start()
    mythread12.start()
    mythread13.start()
    mythread14.start()
    mythread15.start()
    mythread16.start()
    mythread17.start()
    mythread18.start()
    mythread19.start()
    mythread20.start()
    mythread21.start()

if __name__ == "__main__":
    Main()
    stop_time = time.time()
    print("Time taken {}".format(stop_time - start_time))

