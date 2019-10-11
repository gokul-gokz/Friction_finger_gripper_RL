import json

with open('policy.txt') as json_file:
    Q = json.load(json_file)
    


print(Q.keys())
thetas=[-90,0,90]
for i in range(70,110):
    for j in range(70,110):
        for theta in thetas:
            if str((i/10.0,j/10.0,theta)) in Q.keys():
                
                continue
            else:
                print((i/10.0,j/10.0,theta))
