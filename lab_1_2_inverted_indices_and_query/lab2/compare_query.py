# -*- coding: utf-8 -*-

with open("./4_standard.txt", "r") as f_standard:  
    data_standard = f_standard.read()  
    lines_standard=data_standard.split('\n')
    with open("./4_generated.txt", "r") as f_generated:  
        data_generated = f_generated.read()  
        lines_generated=data_generated.split('\n')
        #compare
        same_flag=True
        for f1 in range(0,len(lines_standard)):
            if lines_standard[f1]!=lines_generated[f1]:
                print('standard '+lines_standard[f1])
                print('generated '+lines_generated[f1])
                same_flag=False
                raise Exception(print(str(f1+1)+' conflict'))
        if same_flag==True:
            print('same')