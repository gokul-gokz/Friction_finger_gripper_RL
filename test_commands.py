import random
import json
import numpy as np
from collections import defaultdict

print(round(10.25,1))
print(random.randrange(70,90)/10)

# a=dict()

a=defaultdict(lambda: [0,0])
a['(1,2)'][0]=3
a['(2,3)'][0]=4
print(len(list(range(100, 200))))

with open('Q_table.txt') as json_file:
    Q = json.load(json_file)

print(Q[str((8.3, 10.4, 0))][1])
a='fff'
if(a=='fff'):
    print("yes")
# with open('person.txt', 'w') as json_file:
#   json.dump(a, json_file)
#
# with open('person.txt') as json_file:
#     data = json.load(json_file)
#     print(data['(1,2)'])
#
# a=str((1,2))
#
# print(type(a))
