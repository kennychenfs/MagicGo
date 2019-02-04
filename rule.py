class textcolor:
	bg   ="\x1b[48;5;"
	color="\x1b[38;5;"
	end  ="m"
	reset="\x1b[0m"
step=[[1,0],[-1,0],[0,1],[0,-1]]
looked=[]
def look_init():
	global looked
	looked=[]
	for i in range(13):
		looked.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
class board:
	def __init__(self):
		self.grid=[]
		self.nowturn=1
		self.pregrid=[]
		self.precolor=0
		for i in range(13):
			self.grid.append([])
			for j in range(13):
				self.grid[i].append(0)
	def dump(self,tp=0):
		if tp:
			a='   0 1 2 3 4 5 6 7 8 9101112\n'
			for x in range(13):
				if(len(str(x))==1):
					a+=' '+str(x)
				else:
					a+=str(x)
				for y in range(13):
					if self.grid[x][y]==0:
						if x==3 or x==9:
							if y==3 or y==9:
								a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25CE '+textcolor.reset
								continue
						if x==6 and y==6:
							a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25CE '+textcolor.reset
							continue
						a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+'  '+textcolor.reset
					elif self.grid[x][y]==1:
						a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25cf '+textcolor.reset
					else:
						a=a+textcolor.color+'255'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25cf '+textcolor.reset
				a=a+'\n'
		else:
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
						a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+' '+textcolor.reset
					if self.grid[x][y]==0:
						if x==3 or x==9:
							if y==3 or y==9:
								a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+'+ '+textcolor.reset
								continue
						if x==6 and y==6:
							a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+'+ '+textcolor.reset
							continue
						a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+'. '+textcolor.reset
					elif self.grid[x][y]==1:
						a=a+textcolor.color+'16'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25cf '+textcolor.reset
					else:
						a=a+textcolor.color+'255'+textcolor.end+textcolor.bg+'172'+textcolor.end+u'\u25cf '+textcolor.reset
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
		if looked[x][y]==1:
			return guess
		looked[x][y]=1
		if self.grid[x][y]!=0:
			return self.grid[x][y]
		for i in step:
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
		for i in step:
			if x+i[0]<13 and x+i[0]>=0 and y+i[1]<13 and y+i[1]>=0 and self.grid[x+i[0]][y+i[1]]==0:
				self.fill(x+i[0],y+i[1],color)
	def final(self):
		for x in range(13):
			for y in range(13):
				if self.grid[x][y]==0:
					look_init()
					c=self.area(x,y,None)
					if c==0 or c==None:
						continue
					else:
						self.fill(x,y,c)
		self.dump()
		w=0.0
		b=0
		for x in range(13):
			for y in range(13):
				if self.grid[x][y]==1:
					b+=1
				elif self.grid[x][y]==2:
					w+=1
		w+=6.5
		if(w>b):
			return 2
		elif(b>w):
			return 1
		else:
			return 0
	def ifbreathe(self,x,y,color):#Before using this function, you have to look_init().
		if(self.grid[x][y]==0 and looked[x][y]==0):
			return 1
		looked[x][y]=1
		if(self.grid[x][y]==(not(color-1))+1):
			return 0
		for i in step:
			tx=x+i[0]
			ty=y+i[1]
			if tx>12 or tx<0 or ty>12 or ty<0 or looked[tx][ty]:
				continue
			if self.ifbreathe(tx,ty,color):
				return 1
		return 0
	def play(self,x,y,color):
		look_init()
		if not self.ifbreathe(x,y,color):
			return
		self.grid[x][y]=color
		for i in step:
			look_init()
			if x+i[0]>=0 and y+i[1]>=0 and x+i[0]<13 and y+i[1]<13 and self.grid[x+i[0]][y+i[1]]==(not(color-1))+1 and not self.ifbreathe(x+i[0],y+i[1],(not(color-1))+1):
				self.zero(x+i[0],y+i[1],(not(color-1))+1)
	def zero(self,x,y,color):
		self.grid[x][y]=0
		for i in step:
			tx=x+i[0]
			ty=y+i[1]
			if self.grid[tx][ty]==color:
				self.zero(tx,ty,color)
	def superko(self):
		return
a=board()
a.play(1,1,1)
a.play(2,2,2)
a.play(3,3,1)
a.dump(1)
a.dump(0)
