from rule import board
from random import randint
import datetime
import os
a=input('how many sgf do you want?')
clear=input('Do you want to reset test.sgf?(1=yes,0=no)')
if clear:
	os.remove("test.sgf")
myfile=open("test.sgf", "a")
i=0
now = datetime.datetime.now()
from time import time
start=time()
title='(;GM[1]FF[4]CA[UTF-8]AP[Magic Go]KM[6.5]SZ[13]DT['+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+']PB[MG]PW[MG]'
while i<a:
	game=''
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
		if color is 1:
			game=game+'B['
		else:
			game=game+'W['
		game=game+chr(ord('a')+j%13)+chr(ord('a')+j/13)
		if count!=299:
			game=game+'];'
		else:
			game=game+'])\n'
		count+=1
	i+=1
	score=b.final()
	if score>0:
		game=title+'RE[B+'+str(score)+'];'+game
	else:
		game=title+'RE[w+'+str(-score)+'];'+game
	myfile.write(game)
print (time()-start)/a
