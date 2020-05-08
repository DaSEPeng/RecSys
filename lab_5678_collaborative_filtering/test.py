

from rec_base import *


user_movies_num = load_obj("user_movies_num")



print (len(user_movies_num))

for k,v in user_movies_num.items():
    print(k,"*"*10,v)