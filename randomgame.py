from rule import board
from random import randint
import os
def makegame(num):
	myfile=open("random.mg", "w")
	i=0
	while i<num:
		if i%100==0:
			print i/100,'%'
		b=board(6.5)
		count=0
		passes=0
		color=2
		while count<300:
			color=(not(color-1))+1
			notok=[]
			j=randint(0,170)
			while j in notok or not b.ok(j/13,j%13,color):
				notok.append(j)
				j=randint(0,170)
			b.play(j/13,j%13,color)
			myfile.write(str(j)+' ')
			count+=1
		i+=1
		score=b.final()
		myfile.write(str(score)+'\n')
