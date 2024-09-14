from cmath import isnan
import os
import urllib.request
import pandas as pd



url = 'https://www.creatis.insa-lyon.fr/EvaluationPlatform/picmus/dataset/'

web_url = urllib.request.urlopen(url) 
tables = pd.read_html(web_url.read(), flavor='html5lib')
links = tables[0]


#df[df['A'].str.contains("hello")]



for row in links.iterrows():
    
    vals = row[1]
        
    if str(vals['Name']) is not None and 'iq' in str(vals['Name']):
        x=2
    pass


print("get url done ...")