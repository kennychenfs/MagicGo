#I'm too lazy to type annotation in Chinese.
from rule import board
from random import randint
from copy import deepcopy
class tree:
	under=[]#son nodes, so it's a list
	on=[]#father node
	used=[]#used, I use list to save search result, so I'll keep it after being unvaluable.
	x=[]
	y=[]
	color=[]
	win=[]#win point of black,add for every son node,isn't average
	times=[]#all search times
	nowboard=board(6.5)#not for searching
	def appendnode(self):
		self.x.append(None)
		self.y.append(None)
		self.color.append(None)
		self.on.append(None)
		self.under.append([])
		self.used.append(0)
		self.win.append(None)
		self.times.append(0)
	def expand(self,getf,nowlocal,board,turn):#extand leaf node, now local is where the board it is#f is a function, which returns black win point
		for x in range(13):
			for y in range(13):
				if board.ok(x,y,turn):
					i=0
					while self.used[i]:#find available place to replace
						if i==len(self.under)-1:
							self.appendnode()
						i+=1
					self.x[i]=x
					self.y[i]=y
					self.color[i]=turn
					self.on[i]=nowlocal
					self.under[nowlocal].append(i)
					self.under[i]=[]
					self.used[i]=1
					self.win[i]=getf(board,turn)-board.komi
					self.times[i]=1
		i=0
		while self.used[i]:#find available place to replace
			if i==len(self.under)-1:
				self.appendnode()
			i+=1
		self.x[i]=13
		self.y[i]=13
		self.color[i]=None
		self.on[i]=nowlocal
		self.under[nowlocal].append(i)
		self.under[i]=[]
		self.used[i]=1
		self.win[i]=getf(board,(not(turn-1))+1)-board.komi
		self.times[i]+=1
	def search(self,getf,board,inl,d=0):#inl is input local, the list index
		nowlocal=inl
		while self.under[nowlocal]!=[]:
			for i in self.under[nowlocal]:
				qu=[]#q+u
				if self.times[i]==0:
					print i
				qu.append(self.win[i]/self.times[i]+self.times[nowlocal]/self.times[i]*0.4)#The lastest numder is c_puct
				if d:
					print self.times[i]
			if d:
				print qu
				print len(qu)
			nowlocal=self.under[nowlocal][qu.index(max(qu))]
			if(self.x[nowlocal]!=13):#if this node isn't pass
				board.play(self.x[nowlocal],self.y[nowlocal],self.color[nowlocal])
		self.expand(getf,nowlocal,board,(not(self.color[nowlocal]-1))+1)#expand
		while nowlocal!=inl:
			self.win[nowlocal]=0#here backup start(init)
			self.times[nowlocal]=0
			for i in self.under[nowlocal]:
				self.win[nowlocal]+=self.win[i]
				self.times[nowlocal]+=self.times[i]
			nowlocal=self.on[nowlocal]
ti=0
t=tree()
b=board(6.5)
t.appendnode()
t.used[0]=1
t.color[0]=2
for i in range(100):
	t.search(lambda x,y:7.5,b,0)
