from rule import board
import numpy as np
import randomgame
from copy import deepcopy as dc
randomgame.makegame(100)
myfile=open("random.mg", "r")
a=myfile.readlines()
out=[]
results=[]
for i in a:
	out.append([int(j) for j in i.split()[0:-1]])
	results.append(float(i.split()[-1]))
b=board(6.5)
indata=[]
resultdata=[]
chancedata=[]
for index in range(len(out)):
	agame=out[index]
	b.grid=[]
	for i in range(13):
		b.grid.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
	color=1
	for coor in agame:#coordinate
		b.play(coor/13,coor%13,color)
		indata.append([])
		for i in b.grid:
			indata[-1].append([])
			for j in i:
				indata[-1][-1].append([int(i==1),int(i==2),int(color==1),int(color==2)])
		if (results[index]>0 and color==1) or (results[index]<0 and color==2):
			chancedata.append([])
			for _ in range(170):
				if _==coor:
					chancedata[-1].append(1.0)
				else:
					chancedata[-1].append(0.0)
		else:
			chancedata.append([])
			for _ in range(170):
				if _==coor:
					chancedata[-1].append(0.0)
				else:
					chancedata[-1].append(1.0/168)
		resultdata.append(results[index])
		b.play(coor/13,coor%13,color)
		color=(not(color-1))+1
a=np.array(indata)
b=np.array(chancedata)
c=np.array(resultdata)
np.save('randomdata',a)
np.save('randomdatachance',b)
np.save('randomdataresult',c)
print a.shape
print b.shape
print c.shape
