import itertools

memorylength=5

def get_path(N, max_up, n_up, n_diff, i, all_paths):
    if n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "down"
        all_paths.append(N.copy())
        return
    elif len(N) - n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "up"
        all_paths.append(N.copy())
        return
    if n_diff == 0:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
    else:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
        N[i] = "down"
        get_path(N, max_up, n_up, n_diff-1, i+1, all_paths)
        
for max_up in [memorylength]:
    all_paths = []
    dyckwords=[]
    # max_up = 8
    get_path(["" for i in range(2*max_up)], max_up, 0, 0, 0, all_paths)
    print(f"max_up {max_up}, num_path {len(all_paths)}")
    for path in all_paths:
        print_path = []
        for element in path:
            if element == "up":
                print_path.append("1")
            else:
                print_path.append("0")
        dyckwords.append(print_path)
    print(dyckwords)

def dycktocor(path):
    cl=0
    height=0
    set=[]
    x=0
    count=1
    indic=""
    for i in enumerate(path):
        if i[1]=='1':
            cl+=int(i[1])
            height+=1
            count=1
            
        else:
            if count==1:
                set.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                ss=list(itertools.combinations(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)),2))
                x+=height
                indic=indic+f"{int(i[0]/2-height/2)}{int(i[0]/2+height/2)},"
                #print(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)))
                #print(ss)
                for j in ss:
                    k=j[1]-j[0]
                    if j[1]-j[0]<height:
                        if f"I^{k}_(x_{int(j[0])}x_{int(j[1])})" not in set:
                            #set.append(f"I^{k}_(x_{int(j[0])}x_{int(j[1])})")
                            indic=indic+f"{int(j[0])}{int(j[1])},"
                cl=0
            height+=-1
            count=0
    set.append(x)
    #set.append(indic)
    return(set)

for i in dyckwords:
    print(dycktocor(i))
        
        
        