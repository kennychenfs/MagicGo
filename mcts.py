from rule import board
from random import randint
from copy import deepcopy as dc
import numpy as np
from keras.models import load_model
class node:
	def __init__(self,on,x,y,color):
		self.on=on
		self.w=0.0
		self.n=0
		self.q=0.0
		self.under=[]
		if x==13 and y==13:
			self.passes=1
			self.color=0
		else:
			self.color=color
			self.x=x
			self.y=y
	def select(self,b):
		while(self.under!=[]):
			qu=[]
			if(self.color==1):
				for i in range(len(self.under)):
					qu.append(4.0*self.p[i]*np.sqrt(self.n)/(1+self.under[i].n)+self.under[i].q)
			else:
				for i in range(len(self.under)):
					qu.append(4.0*self.p[i]*np.sqrt(self.n)/(1+self.under[i].n)-self.under[i].q)
			maxx=-1000.0
			index=0
			for i in range(len(qu)):
				if qu[i]>maxx:
					maxx=qu[i]
					index=i
			self=self.under[index]
			b.play(self.x,self.y,self.color)
		return self
	def expand(self,get,b,color):
		self.p,self.w=get(b,color)
		self.p=self.p[0]
		self.w=self.w[0][0]
		for i in range(169):
			self.under.append(node(self,i//13,i%13,(not(color-1))+1))
		self.under.append(node(self,13,13,0))#pass
		return self.w
	def backup(self,v):
		while(self.on!=0):
			self.w+=v
			self.n+=1
			self.q=self.w/self.n
			self=self.on
		self.n+=1
	def mcts(self,b):
		tmpb=dc(b)
		self=self.select(tmpb)
		self.backup(self.expand(resnet_predict,tmpb,self.color))
def resnet_predict(b,turn):#board
	grid=np.array(b.grid)
	x=[[(i==1),(i==2),turn==1,turn==2] for j in grid for i in j]
	x=np.array(x).reshape((1,13,13,4)).astype('float32')
	return model.predict(x)
b=board(6.5)
model=load_model('first.h5')
n=node(0,13,13,2)
for _ in range(100):
	n.mcts(b)
for i in n.under:
	print i.n,i.q*169

