Anmol

Hii
I tried to implement multiple linear regression for 2007 data only. Find the file aarzoo.py for the same.

2007_P is the pollution data and 2007_W is weather data. I have taken the average of values where there 
were multiple rows of same date to make a single row of a date.

# d3 = pd.merge(d2, d1, on="Date")
This is used to combine those rows of both data sets that have same date.

But I am getting vast difference in predicted and expected values.
