# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Clean_Date.py
# Time       ：2023/4/14 11:02
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import pandas as pd
import datetime


# define a function to clean up date columns
def clean_date(date_str):
    # convert the string to a datetime object
    date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d")

    # check if the year is incorrect (e.g. 2023 instead of 2022)
    if date_obj.year == 2023:
        # subtract one year from the date
        date_obj = date_obj.replace(year=date_obj.year - 2)

    # format the date as YYYYMMDD and return as string
    return date_obj.strftime("%Y%m%d")


# read in the original comment data CSV file
comments_df = pd.read_csv("Data/DFCF_SH_2021.csv")

# clean up the "posting time" column
comments_df["PostingTime"] = comments_df["PostingTime"].apply(clean_date)

# read in the SSE000001 CSV file
ssec_df = pd.read_csv("Data/000001SH_Trad.csv")

# clean up the "DateTime" column
ssec_df["DateTime"] = ssec_df["DateTime"].apply(clean_date)

# filter the comments based on SSE000001 trading dates
ssec_dates = set(ssec_df["DateTime"])
comments_df = comments_df[comments_df["PostingTime"].isin(ssec_dates)]

# save the cleaned data to a new CSV file
comments_df.to_csv("Data/DFCF_Date_Clean_2021.csv", index=False, encoding='utf-8-sig', sep=',')