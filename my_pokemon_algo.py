import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pokemon = requests.get('https://pokeapi.co/api/v2/pokemon?offset=0&limit=1126').json()
pokemon_results=pokemon['results']

p = []
for i in pokemon_results:
    test=requests.get(i['url']).json()
    p.append(
        {
            'species': test['species']['name'],
            'height': test['height'],
            'weight':test['weight'],
            'hp':test['stats'][0]['base_stat'],
            'attack':test['stats'][1]['base_stat'],
            'defense':test['stats'][2]['base_stat'],
            'special-attack':test['stats'][3]['base_stat'],
            'special-defense':test['stats'][4]['base_stat'],
            'speed':test['stats'][5]['base_stat'],
            'type':  test['types'][0]['type']['name']
            
        }
    )

dft_all=pd.DataFrame(p)

class_labels = set(list(dft_all['type']))
class_labels_list=list(class_labels)
class_labels_list.sort()

class_dict={}
v=0
for i in class_labels_list:
  class_dict[i]=v
  v+=1


for i in class_dict.keys():
  dft_all.loc[dft_all.type == i, 'type_num'] = int(class_dict[i])

dft_all_fin = dft_all.astype({"type_num": int})

pipe = Pipeline([('scaler', StandardScaler()),
                    ('lasso', LogisticRegression(C=1e6, multi_class="multinomial", solver="lbfgs"))
            ])  


X, y = dft_all_fin[['height','weight','hp','attack','defense','special-attack','special-defense','speed']], dft_all_fin['type_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
preds_test = pipe.predict(X_test)
preds_train = pipe.predict(X_train)
class_labels = class_labels_list





