import pandas as pd


## Load the json files to be converted
# json_file = 'vqa_base_train.json'
# csv_file = 'vqa_base_train.csv'
json_file = 'vqa_base_test.json'
csv_file = 'vqa_base_test.csv'

data = pd.read_json(path_or_buf=json_file, orient='records')

data['img_ind'] = data['img_path']
data.drop(['img_path'],axis=1, inplace= True)

print(data.head())

print(data.shape)

data.to_csv(path_or_buf=csv_file)

