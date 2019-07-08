from copy import deepcopy
class board:
	bg   ="\x1b[48;5;"
	color="\x1b[38;5;"
	end  ="m"
	reset="\x1b[0m"
	def __init__(self,komi):
		self.looked=[]
		self.grid=[]
		self.pregrid=[]
		self.keepeat=0
		self.step=[[1,0],[-1,0],[0,1],[0,-1]]
		self.komi=komi
		for i in range(13):
			self.grid.append([])
			for j in range(13):
				self.grid[i].append(0)
		self.look_init()
	def look_init(self):
		self.looked=[]
		for i in range(13):
			self.looked.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
	def dump(self,tx=13,ty=13):
		black='16';white='255';none='172'
		a='    0 1 2 3 4 5 6 7 8 9101112\n'
		a+=u'  \u250F\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513\n'
		for x in range(13):
			if(len(str(x))==1):
				a+=' '+str(x)
			else:
				a+=str(x)
			a=a+u'\u2503'
			for y in range(13):
				if y==0:
					a=a+self.color+black+self.end+self.bg+none+self.end+' '+self.reset
				if x is tx and y is ty:
					black='196';white='201'
				else:
					black='16';white='255'
				if self.grid[x][y]==0:
					if x==3 or x==9:
						if y==3 or y==9:
							a=a+self.color+black+self.end+self.bg+none+self.end+'+ '+self.reset
							continue
					if x==6 and y==6:
						a=a+self.color+black+self.end+self.bg+none+self.end+'+ '+self.reset
						continue
					a=a+self.color+black+self.end+self.bg+none+self.end+'. '+self.reset
				elif self.grid[x][y]==1:
					a=a+self.color+black+self.end+self.bg+none+self.end+u'\u25cf '+self.reset
				else:
					a=a+self.color+white+self.end+self.bg+none+self.end+u'\u25cf '+self.reset
			a=a+u'\u2503'
			if(len(str(x))==1):
				a+=' '+str(x)
			else:
				a+=str(x)
			a+='\n'
		a+=u'  \u2517\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u251B\n'
		a+='    0 1 2 3 4 5 6 7 8 9101112\n'
		print (u'{}'.format(a))
	def area(self,x,y,guess):
		if self.looked[x][y]==1:
			return guess
		self.looked[x][y]=1
		if self.grid[x][y]!=0:
			return self.grid[x][y]
		for i in self.step:
			if x+i[0]<0 or y+i[1]<0 or x+i[0]>12 or y+i[1]>12:
				continue
			a=self.area(x+i[0],y+i[1],guess)
			if a==0:
				return 0
			else:
				if guess==None:
					guess=a
				else:
					if guess!=a:
						return 0
		return guess
	def fill(self,x,y,color):
		self.grid[x][y]=color
		for i in self.step:
			if x+i[0]<13 and x+i[0]>=0 and y+i[1]<13 and y+i[1]>=0 and self.grid[x+i[0]][y+i[1]]==0:
				self.fill(x+i[0],y+i[1],color)
	def final(self):
		for x in range(13):
			for y in range(13):
				if self.grid[x][y]==0:
					self.look_init()
					c=self.area(x,y,None)
					if c==0 or c==None:
						continue
					else:
						self.fill(x,y,c)
		w=self.komi
		b=0
		for x in range(13):
			for y in range(13):
				if self.grid[x][y]==1:
					b+=1
				elif self.grid[x][y]==2:
					w+=1
		return b-w
	def breathe(self,x,y,color):
		if(self.grid[x][y]==0 and self.looked[x][y]==0):
			return 1
		self.looked[x][y]=1
		if(self.grid[x][y]==(not(color-1))+1):
			return 0
		for i in self.step:
			tx=x+i[0]
			ty=y+i[1]
			if tx>12 or tx<0 or ty>12 or ty<0 or self.looked[tx][ty]:
				continue
			if self.breathe(tx,ty,color):
				return 1
		return 0
	def ifbreathe(self,x,y,color):
		b=0
		if self.grid[x][y]==0:
			self.grid[x][y]=color
			b=1
		elif self.grid[x][y]==(not(color-1))+1:
			print('ifbreathe error')
			return 0
		self.look_init()
		a=self.breathe(x,y,color)
		if b:
			self.grid[x][y]=0
		return a
	def play(self,x,y,color):
		if color==0 or x>=13 or x<0 or y>=13 or y<0:
			return
		tmpboard=deepcopy(self.grid)
		self.grid[x][y]=color
		ifeat=0
		for i in self.step:
			if x+i[0]>=0 and y+i[1]>=0 and x+i[0]<13 and y+i[1]<13 and self.grid[x+i[0]][y+i[1]]==(not(color-1))+1 and not self.ifbreathe(x+i[0],y+i[1],(not(color-1))+1):
				self.zero(x+i[0],y+i[1],(not(color-1))+1)
				ifeat=1
				self.keepeat=1
		if not ifeat:
			self.pregrid=[]
		if ifeat:
			self.pregrid.append(tmpboard)
	def zero(self,x,y,color):
		self.grid[x][y]=0
		for i in self.step:
			tx=x+i[0]
			ty=y+i[1]
			if tx>=0 and ty>=0 and tx<13 and ty<13 and self.grid[tx][ty]==color:
				self.zero(tx,ty,color)
	def ok(self,x,y,color):
		if x>=13 or y>=13 or x<0 or y<0:
		#	print 'pass at rule.py'
			return 1#out of the board means pass
		if self.grid[x][y]!=0:
			return 0
		if self.ifbreathe(x,y,color):
			return 1
		tmpboard=deepcopy(self)
		tmpboard.grid[x][y]=color
		a=0
		for i in self.step:
			if x+i[0]>=0 and y+i[1]>=0 and x+i[0]<13 and y+i[1]<13 and tmpboard.grid[x+i[0]][y+i[1]] == (not(color-1))+1 and (not tmpboard.ifbreathe(x+i[0],y+i[1],(not(color-1))+1)):
				a=1
				tmpboard.zero(x+i[0],y+i[1],(not(color-1))+1)
		if a:
			for i in self.pregrid:
				if tmpboard.grid==i:
				#	print 'see line 168 in rule.py',tmpboard.grid,i
					return 0
			return 1
		return 0
