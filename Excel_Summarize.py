import numpy as np
import os
import csv
import openpyxl as xl

path_root = input('input folder root: (Ex: D:\OA\牛\WA02 2015 9\\799)')
counter = 0
find_name = "flowdata.csv"
content_list=[]

#===========savename==========
print("Total amount:")
print(counter)

sym_num = 0
for i,j in enumerate(path_root):
    if j=="\\":
        sym_num += 1

sym_which = 0
save_name_flag = False
cow_name_flag = False
save_name = ''
cow_name = ''

for i,j in enumerate(path_root):
    if j=="\\":
        sym_which += 1

    if (sym_which == (sym_num-1)):
        save_name_flag = True
    else:
        save_name_flag = False

    if (sym_which == (sym_num)):
        cow_name_flag = True
    else:
        cow_name_flag = False

    if save_name_flag and j!="\\":
        save_name = save_name + j
    if cow_name_flag and j!="\\":
        cow_name = cow_name + j

save_name_modi = save_name.replace(" ","_")
save_name_modi = cow_name + save_name_modi + ".csv"
print(save_name_modi)

#============collect data=================
for root, dirs, files in os.walk(path_root):
    for d in dirs:
        print(d)
        path_list = os.path.join(root, d)
        flag = 0
        for filename in os.listdir(path_list):
            if (find_name == filename):
                file_path = path_list + '\\' + filename

                with open(file_path, "rt") as csvfile:
                    reader = csv.reader(csvfile)
                    for i, rows in enumerate(reader):
                        if i == 1:
                            content = rows

                content.append(cow_name)
                print(content)
                content_list.append(content)

                flag = 1
                counter+=1
                break
        if(flag==0):
            warning_message = 'no flowdata.csv in ' + path_list
            print(warning_message)


#=========save==============
try:
    exc_save_magn = path_root + '\\' + save_name_modi

    book = xl.Workbook()
    sheet = book.active
    for i in range (counter):
        sheet.append(content_list[i])
    book.save(exc_save_magn)
    print("Saved")

except:
    print("Falied!!!")

con = input('>>')






