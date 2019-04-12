from rule import board
from random import randint
from copy import deepcopy as dc
import numpy as np
from keras.models import load_model
from time import time
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
			if(self.color==2):
				qu=[4.0*self.p[i]*np.sqrt(self.n)/(1+self.under[i].n)+self.under[i].q for i in range(len(self.under))]
			else:
				qu=[4.0*self.p[i]*np.sqrt(self.n)/(1+self.under[i].n)-self.under[i].q for i in range(len(self.under))]
			index=qu.index(max(qu))
			self=self.under[index]
			if self.x!=13:
				b.play(self.x,self.y,self.color)
		return self
	def expand(self,get,b,color):
		self.p,self.w=get(b,color)
		self.p=self.p[0]
		self.w=self.w[0][0]
		for i in range(169):
			if b.ok(i//13,i%13,color):
				self.under.append(node(self,i//13,i%13,color))
		self.under.append(node(self,13,13,0))#pass
		return self.w
	def backup(self,v):
		while(self.on!=0):
			self.w+=v
			self.n+=1
			self.q=self.w/self.n
			self=self.on
		self.n+=1
		return self
	def mcts(self,b):
		tmpb=dc(b)
		self=self.select(tmpb)
		self.backup(self.expand(resnet_predict,tmpb,(not(self.color-1))+1))
def resnet_predict(b,turn):#board
	grid=np.array(b.grid)
	x=[[(i==1),(i==2),turn==1,turn==2] for j in grid for i in j]
	x=np.array(x).reshape((1,13,13,4)).astype('float32')
	return model.predict(x)
b=board(6.5)
model=load_model('first.h5')
n=node(0,13,13,2)
start=time()
tmpb=dc(b)
for _ in range(1000):
	self=n.select(tmpb)
	self=self.backup(self.expand(resnet_predict,tmpb,(not(self.color-1))+1))
	tmpb.grid=dc(b.grid)
print 'took:',time()-start,'s'
self=n.select(tmpb)
self=self.backup(self.expand(resnet_predict,tmpb,(not(self.color-1))+1))
for i in n.under:
	print i.n,i.q*169
