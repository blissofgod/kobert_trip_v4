#with open("./data/cost/temp.txt", "r",encoding="utf-8") as file:
 #   for line in file:
  #      line=line.replace('\n','')
   #     print(line + "    2")

import re
with open("./data/cost/temp.txt", "r",encoding="utf-8") as file:
    for line in file:

        line = line.replace('.','')
        line = line.replace('\n','')
        line = line.replace('     ','')
        line = line.replace('B-','')

        print(line)


