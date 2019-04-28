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
		if x==13 and y==0:
			self.passes=1	
		self.color=color
		self.x=x
		self.y=y
	def select(self,b):
		while(self.under!=[]):
		#	print 'color',self.color
			if(self.color==2):
				qu=[0]*170#init a list:[0,0,...,0]
				for i in self.under:
					index=i.x*13+i.y
					qu[index]=2.0*self.p[index]*np.sqrt(self.n)/(1+i.n)+i.q#pass will be (x:13,y:0)
				qu=[2.0*self.p[i]*np.sqrt(self.n) if qu[i] is 0 else qu[i] for i in range(170)]
			else:
				qu=[0]*170#init a list:[0,0,...,0]
				for i in self.under:
					index=i.x*13+i.y
					qu[index]=2.0*self.p[index]*np.sqrt(self.n)/(1+i.n)-i.q#pass will be (x:13,y:0)
				qu=[2.0*self.p[i]*np.sqrt(self.n) if qu[i] is 0 else qu[i] for i in range(170)]
			index=qu.index(max(qu))
			while((not index is 170) and (not b.ok(index/13,index%13,(not(self.color-1))+1))):
			#	b.dump()
			#	print index/13,index%13,'can\'t be played.'
				qu[index]=-1000.0
				index=qu.index(max(qu))
			a=-1#if a==-1,the step is new
				#else this step is a
			for i in self.under:
				if i.x==index/13 and i.y==index%13:
					a=i
			if a==-1:#this step doesn't exist in self.under
				self.under.append(node(self,index/13,index%13,(not(self.color-1))+1))#new a node
				if index!=170:
					b.play(index/13,index%13,(not(self.color-1))+1)
				return self.under[-1]
			self=a
			if self.x!=13:
				b.play(self.x,self.y,self.color)
		qu=[2.0*self.p[i]*np.sqrt(self.n) for i in range(170)]
		index=qu.index(max(qu))
		self.under.append(node(self,index/13,index%13,(not(self.color-1))+1))
		self=self.under[0]
		if self.x!=13:
			b.play(self.x,self.y,self.color)
		return self
	def expand(self,get,b,color):
		self.p,self.w=get(b,color)
		self.p=self.p[0]
		self.w=self.w[0][0]
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
model=load_model('low.h5')
n=node(0,13,13,2)
start=time()
tmpb=dc(b)
n.expand(resnet_predict,tmpb,1)
for _ in range(1600):
	self=n.select(tmpb)
	self=self.backup(self.expand(resnet_predict,tmpb,(not(self.color-1))+1))
	tmpb.grid=dc(b.grid)
print 'took:',time()-start,'s'
self=n.select(tmpb)
self=self.backup(self.expand(resnet_predict,tmpb,(not(self.color-1))+1))
for i in n.under:
	print i.n,i.q*169
