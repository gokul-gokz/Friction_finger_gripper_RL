import json

with open('Valid_action_table.txt') as json_file:
    valid_Actions = json.load(json_file)

# new_policy=dict()
# for key in valid_Actions.keys():
#     set=[]
#     for action in valid_Actions[key]:
#         if action==0:
#             if 'r' not in set:
#                 set.append('r')
#         elif action==1:
#             if 'l' not in set:
#                 set.append('l')
#         elif action==2:
#             if 'l' not in set:
#                 set.append('l')
#         elif action==3:
#             if 'r' not in set:
#                 set.append('r')
#         elif action == 4:
#             if 'l' not in set:
#                 set.append('l')
#         elif action == 5:
#             if 'r' not in set:
#                 set.append('r')
#     set.append('hh')
#     set.append('lh')
#     set.append('hl')
#     new_policy[key]=set
#
# with open('New_Valid_action_table.txt','w') as json_file:
#     json.dump(new_policy,json_file)
#

# c=(2,3,'lh')
# d="(2,3)"
# e=d.replace(')',",'lh')")
# print(e)
# print(str(c))
# if(str(c)==e):
#     print('yes')

with open('Policy.txt') as json_file:
    valid_Actions = json.load(json_file)
    for key in valid_Actions.keys():
        if key==str((7.7, 8.8, 0, 'lh')):
            print("yes")
