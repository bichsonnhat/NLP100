# import json
# import pandas as pd

# # create a Python dictionary to store the data
# file = "D:/GitSource/PythonTraining/Chapter 3 - Regular Expression/enwiki-country.json"
# # data = {"name": "John", "age": 30, "city": "New York"}

# # open the JSON file
# with open('D:/GitSource/PythonTraining/Chapter 3 - Regular Expression/enwiki-country.json', "r") as f:
#     json_str = f.read()

# data = json.loads(json_str)

# print(data)

# import json
# import pandas as pd

# df = pd.read_json('states.json')

# print(df)

# for state in data['states']:
#   del state['area_codes']

# with open('new_states.json', 'w') as f:
#   json.dump(data, f, indent=2)

import pandas as pd

# read JSON data from a file into a DataFrame
df = pd.read_json('/Users/Admin/Desktop/enwiki-country.json', encoding = "utf-8")

# print the DataFrame
print(df)
