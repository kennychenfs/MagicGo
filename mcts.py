#coding:utf-8
#參考 https://github.com/chengstone/cchess-zero
#almost the same as that
from rule import board
from random import randint
from copy import copy
import numpy as np
from keras.models import load_model
import time
import asyncio
#mcts
c_puct=5
#resnet
BLOCKS=15
FILTERS=192
is_using_TPU=False
#if I use tpu, optimizer must handle by "tf.tpu.CrossShardOptimizer" to cross more than one tpus.And then use it as GPU!!!
def get_residual_block(input):
	conv=tf.layers.conv2d(inputs=input,filters=FILTERS,kernel_size=(3,3),padding="same",activation=None)
	bn=tf.layers.batch_normalization(inputs=conv,axis=3,center=False,scale=False,fused=True);bn=tf.nn.relu(bn)
	conv=tf.layers.conv2d(inputs=bn,filters=FILTERS,kernel_size=(3,3),padding="same",activation=None)
	bn=tf.layers.batch_normalization(inputs=conv,axis=3,center=False,scale=False,fused=True);bn=tf.nn.relu(bn)
	return tf.nn.relu(tf.add(input,bn))
def resnet_fn(features, labels, mode):
	#The features must be: <tensor, shape=(batch_size, 13, 13, 4)>
	#The labels must be: {'policy':<tensor, shape=(batch_size, 170)>, 'value':<tensor, shape=(batch_size, 1)>}
	#which value is z, and policy is pi in AlphaGo Zero paper
	#not to reshape as possible because TPU doesn't good at it
	
	#init
	layer=tf.layers.conv2d(inputs=features, filters=FILTERS, kernel_size=(3, 3), padding="same", activation=None)
	layer=tf.layers.batch_normalization(inputs=layer, axis=3, center=False, scale=False, fused=True)
	layer=tf.nn.relu(layer)
	#residual block
	for i in range(BLOCKS-1):
		layer=get_residual_block(layer)
	#policy head
	policy_head=tf.layers.conv2d(inputs=layer, filters=2, kernel_size=(1, 1), padding="same", activation=None)
	policy_head=tf.layers.batch_normalization(inputs=policy_head, axis=3, center=False, scale=False, fused=True);policy_head=tf.nn.relu(policy_head)
	policy_head=tf.reshape(policy_head, [-1, 338])#which 676=13*13*2(two filters)
	policy_head_output=tf.layers.dense(inputs=policy_head, units=170) #which 170=13*13+1, and the 1 is passactivation=tf.nn.relu)
	
	value_head=tf.layers.conv2d(inputs=layer, filters=1, kernel_size=(1, 1), padding="same", activation=None)
	value_head=tf.layers.batch_normalization(inputs=value_head, axis=3, center=False, scale=False, fused=True);value_head=tf.nn.relu(value_head)
	value_head=tf.reshape(value_head, [-1, 169])#which 676=13*13*1(one filter)
	value_head=tf.layers.dense(inputs=value_head, units=256, activation=tf.nn.relu)
	value_head_output=tf.layers.dense(inputs=value_head, units=1) # a scalaractivation=tf.math.tanh)
	predictions={'policy': tf.nn.softmax(policy_head_output, name='policy_head'),'value' : tf.identity(value_head_output, name='value_head')}
	if mode is tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	policy_loss=tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy'], logits=policy_head_output)# it does soft max to logits before cross entropy
	#its output is an array, needing to reduce mean to a scalar
	policy_loss=tf.reduce_mean(policy_loss)
	
	value_loss=tf.losses.mean_squared_error(labels=labels['value'], predictions=value_head_output)
	value_loss=tf.reduce_mean(value_loss)
	
	regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4)#as the AGZ paper, c||θ||_2
	regular_variables=tf.trainable_variables()#refers to the trainable things, including all the conv2d, batchnorm, and dense.
	l2_loss=tf.contrib.layers.apply_regularization(regularizer, regular_variables)
	loss=value_loss + policy_loss + l2_loss
	def learning_rate_fn(global_step):
		boundaries=[400, 600]
		learning_rates=[1e-2, 1e-3, 1e-4]
		return tf.train.piecewise_constant(global_step, boundaries, learning_rates)
	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step=tf.train.get_or_create_global_step()
		learning_rate=learning_rate_fn(global_step)
		optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)#continue here
		train_op=optimizer.minimize(loss=loss, global_step=global_step)#which global_step is a variable
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	#no evaluate mode
def resnet_fn(features, labels, mode):
	#The features must be: <tensor, shape=(batch_size, 13, 13, 4)>
	#The labels must be: {'policy':<tensor, shape=(batch_size, 170)>, 'value':<tensor, shape=(batch_size, 1)>}
	#which value is z, and policy is pi in AlphaGo Zero paper
	#not to reshape as possible because TPU doesn't good at it
	
	#init
	layer=tf.layers.conv2d(inputs=features, filters=FILTERS, kernel_size=(3, 3), padding="same", activation=None)
	layer=tf.layers.batch_normalization(inputs=layer, axis=3, center=False, scale=False, fused=True)
	layer=tf.nn.relu(layer)
	#residual block
	for i in range(BLOCKS-1):
		layer=get_residual_block(layer)
	#policy head
	policy_head=tf.layers.conv2d(inputs=layer, filters=2, kernel_size=(1, 1), padding="same", activation=None)
	policy_head=tf.layers.batch_normalization(inputs=policy_head, axis=3, center=False, scale=False, fused=True);policy_head=tf.nn.relu(policy_head)
	policy_head=tf.reshape(policy_head, [-1, 338])#which 676=13*13*2(two filters)
	policy_head_output=tf.layers.dense(inputs=policy_head, units=170) #which 170=13*13+1, and the 1 is passactivation=tf.nn.relu)
	
	value_head=tf.layers.conv2d(inputs=layer, filters=1, kernel_size=(1, 1), padding="same", activation=None)
	value_head=tf.layers.batch_normalization(inputs=value_head, axis=3, center=False, scale=False, fused=True);value_head=tf.nn.relu(value_head)
	value_head=tf.reshape(value_head, [-1, 169])#which 676=13*13*1(one filter)
	value_head=tf.layers.dense(inputs=value_head, units=256, activation=tf.nn.relu)
	value_head_output=tf.layers.dense(inputs=value_head, units=1) # a scalaractivation=tf.math.tanh)
	predictions={'policy': tf.nn.softmax(policy_head_output, name='policy_head'),'value' : tf.identity(value_head_output, name='value_head')}
	if mode is tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	policy_loss=tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy'], logits=policy_head_output)# it does soft max to logits before cross entropy
	#its output is an array, needing to reduce mean to a scalar
	policy_loss=tf.reduce_mean(policy_loss)
	
	value_loss=tf.losses.mean_squared_error(labels=labels['value'], predictions=value_head_output)
	value_loss=tf.reduce_mean(value_loss)
	
	regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4)#as the AGZ paper, c||θ||_2
	regular_variables=tf.trainable_variables()#refers to the trainable things, including all the conv2d, batchnorm, and dense.
	l2_loss=tf.contrib.layers.apply_regularization(regularizer, regular_variables)
	loss=value_loss + policy_loss + l2_loss
	def learning_rate_fn(global_step):
		boundaries=[400, 600]
		learning_rates=[1e-2, 1e-3, 1e-4]
		return tf.train.piecewise_constant(global_step, boundaries, learning_rates)
	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step=tf.train.get_or_create_global_step()
		learning_rate=learning_rate_fn(global_step)
		optimizer = tf.tpu.CrossShardOptimizer(
			tf.train.MomentumOptimizer(learning_rate = learning_rate,momentum = 0.9)
		)
		train_op = optimizer.minimize(loss = loss,global_step = global_step)#which global_step is a variable
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
	#no evaluate mode
class node:
	def __init__(self,parent=0,x,y,p=0,grid=[],depth=0):#action169 is pass
		global c_puct
		self.grid=grid
		if grid is []:
			for _ in range(13):
				self.grid.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
		self.depth=depth
		self.parent=parent
		self.p=p#self
		self.w=0
		self.n=0
		self.q=q
		self.update_u(c_puct)
		self.child={}#{action:children(node)...}
		if type(x) is not int or (x>=13 or y>=13 or x<0 or y<0):
			self.passes=True
		else:
			self.passes=False
		self.x=x
		self.y=y
	def get_q_add_u(self):
		return self.q+self.u
	def select(self):
		move=max(self.child.items(), key=lambda node: node[1].get_q_add_u())
		return move
	def update_u(self,c_puct):
		self.u=c_puct*self.p*sqrt(self.parent.n)/(1+self.n)
	def backup(self,v):
		global c_puct
		self.w+=v
		self.n+=1
		self.q=self.w/self.n
		if self.parent is not 0:
			self.update_u(c_puct)
	def expand(self,moves,P,color):
		total=1e-8
		b = board(grid=self.grid)
		P=P.flatten()#P is a array from tensorflow
		for x,y in moves:
			b.grid=self.grid
			if x<13 and y<13 and x>=0 and y>=0:
				b.play(x,y,color)
			p=P[x*13+y]
			newnode=node(self,x,y,color,p,b.grid,depth=self.depth+1)
			self.child[(x,y)]=newnode
			total+=p
		for i in self.child.values:
			i.p/=total
QueueItem = namedtuple("feature", "future")
class mcts_tree:
	def __init__(self,search_threads,estimator):
		self.root=node(x=None,y=None,color=2,depth=0)
		self.virtual_loss=3
		self.now_expanding=set()
		self.expanded=set()
		self.sem=asyncio.Semaphore(search_threads)
		self.queue=Queue(search_threads)#prediction queue
		self.loop=asyncio.get_event_loop()
		self.running_simulation_num = 0
		self.board=board()
		self.estimator=estimator
	def Q(self, move):
		rate=0.0
		found=False
		for action,child in self.root.child.items():# self.root.child is a dict.
			if move==action:
				rate=child.Q
				found=True
		if not found:
			print("{} not exist in the children".format(move))
		return rate
	def update_tree(self,act):#make act be the root.
		self.expanded.discard(self.root)
		self.root=self.root.child[act]
		self.root.parent=0
	def is_expanded(self,content):
		return content in self.expanded
	async def tree_search(self,nownode,now_turn):
		self.running_prediction_num+=1
		
		with await self.sem:
			value=await self.start_tree_search(nownode, now_turn)
			self.running_prediction_num-=1
			
		return value
	async def start_tree_search(self, nownode, now_turn):
		#output is the V for the now turn player
		now_expanding = self.now_expanding

		while nownode in now_expanding:
			await asyncio.sleep(1e-4)

		if not self.is_expanded(nownode):
			#here is expanding
			self.now_expanding.add(nownode)
			#start expanding
			situation=[[i==1,i==2,now_turn==1,now_turn==2] for j in nownode.grid for i in j]
			future = await self.push_queue(situation)#now it's list
			await future
			action_probs, value = future.result()
			action_probs = action_probs[0]
			value=value[0][0]
			if now_turn is 2:
				value = -value
			i=0
			b=board(grid=nownode.grid)
			moves=[]
			while i<=169:#including 169, pass
				if b.ok(i/13,i%13,now_turn):
					moves.append((i/13,i%13))
				i+=1
			del b
			node.expand(moves, action_probs,now_turn)
			self.expanded.add(nownode)  # nownode.state

			self.now_expanding.discard(nownode)

			#don't need to invert because value is always for black
			return value

		else:
			"""node has already expanded. Enter select phase."""
			# select child nownode with maximum action score
			#last_state = nownode.state

			action,nownode=nownode.select()

			# action_t = self.select_move_by_action_score(key, noise=True)

			# add virtual loss
			# self.virtual_loss_do(key, action_t)
			nownode.N+=virtual_loss
			nownode.W-=virtual_loss

			# evolve game board status
			# child_position = self.env_action(position, action_t)
			
			if (nownode.passes and nownode.parent.passes) or depth is 338:# 338=13*13*2, according to AlphaGoZero Paper Page 22
				self.board.grid = nownode.grid
				value=b.final()/169.0
			else:
				value = await self.start_tree_search(nownode, now_turn)  #遞迴
			nownode.N-=virtual_loss
			nownode.W+=virtual_loss

			# on returning search path
			# update: N, W, Q, U
			# self.back_up_value(key, action_t, value)
			if now_turn is 1:
				#black
				nownode.backup(value)
			else:
				nownode.backup(-value)
			#到目前為止nownode都是被選中的child
			#因為是遞迴，所以除了未展開的節點都會backup
			#函數看到的value一直都是對黑棋來說的子差/169.0
			#但在儲存(backup)時就得把白棋的value變號
			return value
	'''
	def flip_board(self,board):
		direction=randint(0,3)#0,1,2,3
		flip=randint(0,1)#0,1=True,False
		#When flipping, the right upper corner doesn't change.
		if direction is 0:
			if flip:
				return [[board[y][x] for y in range(13)] for x in range(13)]# it should be board[x][y] when not flipping.
			else:
				return board
		elif direction is 1:#clockwise
			if flip:
				return [[board[12-x][y] for y in range(13)] for x in range(13)]
			else:
				return [[board[12-y][x] for y in range(13)] for x in range(13)]
		elif direction is 2:
			if flip:
				return [[board[12-y][12-x] for y in range(13)] for x in range(13)]
			else:
				return [[board[12-x][12-y] for y in range(13)] for x in range(13)]
		else:#anticlockwise
			if flip:
				return [[board[x][12-y] for y in range(13)] for x in range(13)]
			else:
				return [[board[y][12-x] for y in range(13)] for x in range(13)]
	'''
	# I wanted to flip board before prediction, but it's too complicated to handle policy.
	# Use it when the training is blocked.
	
	async def prediction_worker(self):
		"""For better performance, queueing prediction requests and predict together in this worker.
		speed up about 45sec -> 15sec for example.
		"""
		q = self.queue
		margin = 10  # avoid finishing before other searches start.
		#allowing empty q for 'margin' times.
		while self.running_simulation_num > 0 or margin > 0:
			if q.empty():
				if margin>0:
					margin-=1
				await asyncio.sleep(1e-3)#Wait a long time, and there should be some tasks in q.
				continue
			item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]#get all features to predict
			features = [item.feature for item in item_list]#item is QueueItem#features=array[feature,featrue...], each feature is the shape of (13,13,4)
			input_fn = tf.estimator.inputs.numpy_input_fn(x=tf.cast(features,'bfloat16' if is_using_TPU else 'float32'),shuffle=False)
			results = self.estimator.predict(input_fn)
			action_probs, value = results['policy'],results['value']
			#action_probs becomes (xxx,170), value (xxx,1)
			for p, v, item in zip(action_probs, value, item_list):
				item.future.set_result((p, v))
	async def push_queue(self, features):
		future = self.loop.create_future()
		item = QueueItem(features, future)
		await self.queue.put(item)
		return future
	def main(self, board, now_turn, playouts):
		node = self.root
		if not self.is_expanded(node):    # and node.is_leaf()    # node.state
			# print('Expadning Root Node...')
			situation=[[i==1,i==2,now_turn==1,now_turn==2] for j in nownode.grid for i in j]
			input_fn = tf.estimator.inputs.numpy_input_fn(x=features,shuffle=False)
			action_probs, value = self.estimator.predict(input_fn)
			if now_turn is 2:
				value = -value
			i=0
			b=board(grid=nownode.grid)
			moves=[]
			while i<=169:#including 169, pass
				if b.ok(i/13,i%13,now_turn):
					moves.append((i/13,i%13))
				i+=1
			del b
			node.expand(moves, action_probs,now_turn)
			self.expanded.add(nownode)
		now_turn = (not(now_turn-1))+1
		coroutine_list = []
		for _ in range(playouts):
			coroutine_list.append(self.tree_search(nownode, now_turn, restrict_round))
		coroutine_list.append(self.prediction_worker())
		self.loop.run_until_complete(asyncio.gather(*coroutine_list))
class go_main:
	def __init__(self, playout=1600, in_batch_size=512, exploration = True, in_search_threads = 16, processor = "gpu"):
		self.epochs = 5
		self.playout_counts = playout    #400    #800    #1600    200
		#self.temperature = 1    #1e-8    1e-3
		#我不想要用temperature,因為我覺得趨近於零時，就直接選N最大的就好了
		self.batch_size = in_batch_size    #128    #512
		self.start_steps = 30	#when the temperature is 1
		self.start_temperature = 1    #2
		# self.Dirichlet = 0.3    # P(s,a) = (1 - ϵ)p_a  + ϵη_a    #self-play chapter in the paper
		'''
		self.eta = 0.03
		# self.epsilon = 0.25
		# self.v_resign = 0.05
		# self.c_puct = 5
		self.learning_rate = 0.001    #5e-3    #    0.001
		self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
		'''
		self.buffer_size = 10000
		self.data_buffer = deque(maxlen=self.buffer_size)
		self.game_borad = board()
		if is_using_TPU:
			self.policy_value_netowrk = tf.estimator.tpu.TPUEstimator(model_fn=resnet_fn_tpu, model_dir="MGmodels")
		else:
			self.policy_value_netowrk = tf.estimator.tpu.Estimator(model_fn=resnet_fn, model_dir="/home/kenny/Desktop/python3/MG/MGmodels")
		self.search_threads = in_search_threads
		self.mcts=mcts_tree(self.search_threads,self.policy_value_network_gpus)
		self.exploration = exploration
	def selfplay(self):
		self.game_borad.reload()
		# p1, p2 = self.game_borad.players
		states, mcts_probs, current_players = [], [], []
		z = 0.0#z是浮點數，因為要存子差
		game_over = False
		start_time = time.time()
		while(not game_over):
			action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
			state, palyer = self.mcts.try_flip(self.game_borad.state, self.game_borad.now_turn, self.mcts.is_black_turn(self.game_borad.now_turn))
			states.append(state)
			prob = np.zeros(labels_len)
			if self.mcts.is_black_turn(self.game_borad.now_turn):
				for idx in range(len(probs[0][0])):
					# probs[0][0][idx] = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
					act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
					# for idx in range(len(mcts_prob[0][0])):
					prob[label2i[act]] = probs[0][1][idx]
			else:
				for idx in range(len(probs[0][0])):
					prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
			mcts_probs.append(prob)
			# mcts_probs.append(probs)
			current_players.append(self.game_borad.now_turn)

			last_state = self.game_borad.state
			# print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))
			self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
			self.game_borad.round += 1
			self.game_borad.now_turn = "w" if self.game_borad.now_turn == "b" else "b"
			if is_kill_move(last_state, self.game_borad.state) == 0:
				self.game_borad.restrict_round += 1
			else:
				self.game_borad.restrict_round = 0

			# self.game_borad.print_borad(self.game_borad.state, action)

			if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
				z = np.zeros(len(current_players))
				if (self.game_borad.state.find('K') == -1):
					winnner = "b"
				if (self.game_borad.state.find('k') == -1):
					winnner = "w"
				z[np.array(current_players) == winnner] = 1.0
				z[np.array(current_players) != winnner] = -1.0
				game_over = True
				print("Game end. Winner is player : ", winnner, " In {} steps".format(self.game_borad.round - 1))
			elif self.game_borad.restrict_round >= 60:
				z = np.zeros(len(current_players))
				game_over = True
				print("Game end. Tie in {} steps".format(self.game_borad.round - 1))
			# elif(self.mcts.root.v < self.resign_threshold):
			#     pass
			# elif(self.mcts.root.Q < self.resign_threshold):
			#    pass
			if(game_over):
				# self.mcts.root = leaf_node(None, self.mcts.p_, "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
				self.mcts.reload()
		print("Using time {} s".format(time.time() - start_time))
		return zip(states, mcts_probs, z), len(z)
	
