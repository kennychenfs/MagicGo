import numpy as np
import tensorflow as tf
BLOCKS=15
FILTERS=192
def get_residual_block(input):
	conv=tf.layers.conv2d(inputs=input,filters=FILTERS,kernel_size=(3,3),padding="same",activation=None)
	bn=tf.layers.batch_normalization(inputs=conv,axis=3,center=False,scale=False,fused=True);bn=tf.nn.relu(bn)
	conv=tf.layers.conv2d(inputs=bn,filters=FILTERS,kernel_size=(3,3),padding="same",activation=None)
	bn=tf.layers.batch_normalization(inputs=conv,axis=3,center=False,scale=False,fused=True);bn=tf.nn.relu(bn)
	return tf.nn.relu(tf.add(input,bn))
def resnet_fn(features, labels, mode):
	#The features must be: <tensor, shape=(batch_size, 13, 13, 4)>
	#The labels must be: {'policy':<tensor, shape=(batch_size, 170)>, 'value':<tensor, shape=(batch_size)>}
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
size=30000
a=np.random.randint(0, 2, size=(size, 13, 13, 4)).astype('float32')
value=np.random.random_sample(size=(size,1)).astype('float32')*2-1
policy=np.random.random_sample(size=(size, 170)).astype('float32')
total=np.sum(policy,axis=1)
policy=np.asarray([policy[i]/total[i] for i in range(size)])
b=size//1024
c=size//b
a=np.split(a,[i*c for i in range(1,1+b)])
value=np.split(value,[i*c for i in range(1,1+b)])
policy=np.split(policy,[i*c for i in range(1,1+b)])
checkpoint_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 300,  # Save checkpoints every 5 minutes.
    keep_checkpoint_max = 5,       # Retain the 5 most recent checkpoints.
)
estimator=tf.estimator.Estimator(model_fn=resnet_fn, model_dir="/home/kenny/Desktop/python3/MG/MGmodels",config=checkpoint_config)
i=0
for a_b,policy_b,value_b in zip(a,policy,value):
	estimator.train(tf.estimator.inputs.numpy_input_fn(x=a_b,y={'policy':policy_b,'value':value_b},num_epochs=1,shuffle=False),steps=1000)
	i+=1
	print(i)
