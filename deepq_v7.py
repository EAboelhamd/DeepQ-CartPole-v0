# Deep Double Q with Experience Replay

import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPERPARMETERS
H = 15
H2 = 15
batch_number = 50
gamma = 0.99
explore = 1
num_of_episodes_between_q_copies = 50
learning_rate=1e-3

    
    
if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    
    tf.reset_default_graph()

    #First Q Network
    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -.10, .10))
    bias1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    
    w2 = tf.Variable(tf.random_uniform([H,H2], -.10, .10))
    bias2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    
    w3 = tf.Variable(tf.random_uniform([H2,env.action_space.n], -.10, .10))
    bias3 = tf.Variable(tf.random_uniform([env.action_space.n], -.10, .10))
    
    states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="states")  # This is the list of matrixes that hold all observations
    actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="actions")
    
    hidden_1 = tf.nn.relu(tf.matmul(states, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    action_values = tf.matmul(hidden_2, w3) + bias3
    
    Q = tf.reduce_sum(tf.mul(action_values, actions), reduction_indices=1) 
    
    #previous_action_masks = tf.placeholder(tf.float32, [None, env.action_space.n], name="p_a_m") # This holds all actions taken 
    #previous_values = tf.reduce_sum(tf.mul(action_values, previous_action_masks), reduction_indices=1) #Combination of action taken and resulting q
    
    #Is there a better way to do this?
    w1_prime = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

    
    w3_prime = tf.Variable(tf.random_uniform([H2,env.action_space.n], -1.0, 1.0))
    bias3_prime = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))
    
    #Second Q network
    
    next_states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="n_s") # This is the list of matrixes that hold all observations
    hidden_1_prime = tf.nn.relu(tf.matmul(next_states, w1_prime) + bias1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    next_action_values =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime
    
     #need to run these to assign weights from Q to Q_prime
    w1_prime_update= w1_prime.assign(w1)
    bias1_prime_update= bias1_prime.assign(bias1)
    w2_prime_update= w2_prime.assign(w2)
    bias2_prime_update= bias2_prime.assign(bias2)
    w3_prime_update= w3_prime.assign(w3)
    bias3_prime_update= bias3_prime.assign(bias3)
    
    #Q_prime = rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)
    
    #we need to train Q
    one_hot = tf.placeholder(tf.float32, [None, 2], name="training_mask")
    rewards = tf.placeholder(tf.float32, [None, ], name="rewards") # This holds all the rewards that are real/enhanced with Qprime
    loss = tf.reduce_sum(tf.square(rewards - Q)) #* one_hot  
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment
    
    max_episodes = 2000
    max_steps = 199

    D = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        sess.run(init)
        #Copy Q over to Q_prime
        sess.run(w1_prime_update)
        sess.run(bias1_prime_update)
        sess.run(w2_prime_update)
        sess.run(bias2_prime_update)
        sess.run(w3_prime_update)
        sess.run(bias3_prime_update)
    
        for episode in xrange(max_episodes):
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
            reward_sum = 0
            new_state = env.reset()
            
            for step in xrange(max_steps):
                if(step == (max_steps-1)):
                    print 'Made 199 steps!'
                
                #if episode % batch_number == 0:
                env.render()
                
                state = list(new_state);
                
                if explore > random.random():
                    action = env.action_space.sample()
                    if(action == 1.0):
                        curr_action = [0.0,1.0] 
                    else:
                        curr_action = [1.0,0.0]
                else:
                
                    #get action from policy
                    results = sess.run(action_values, feed_dict={states: np.array([new_state])})
                    action = (np.argmax(results[0]))
                    if(action == 1.0):
                        curr_action = [0.0,1.0]
                    else:
                        cur_action = [1.0,0.0]
                
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward
                
                D.append([state, curr_action, reward, new_state, done])
                
                if len(D) > 5000:
                    D.pop(0)
                #Training a Batch
                sample_size = len(D)
                if sample_size > 500:
                    sample_size = 500
                else:
                    sample_size = sample_size
                 
                if True:
                    samples = [ D[i] for i in random.sample(xrange(len(D)), sample_size) ]
                    new_states_for_q = [ x[3] for x in samples]
                    all_q_prime = sess.run(next_action_values, feed_dict={next_states: new_states_for_q})
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        #print i_sample
                        if i_sample[4] == True:
                            y_.append(reward)
                        else:
                            this_q_prime = all_q_prime[ind]
                            maxq = max(this_q_prime)
                            y_.append(reward + (gamma * maxq))
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(i_sample[1])
                    
                    sess.run(train, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples, one_hot: actions_samples})
                
                if done:
                    break
                
            if episode % num_of_episodes_between_q_copies == 0:
                sess.run(w1_prime_update)
                sess.run(bias1_prime_update)
                sess.run(w2_prime_update)
                sess.run(bias2_prime_update)
                sess.run(w3_prime_update)
                sess.run(bias3_prime_update)
            
            explore = explore * .997
            #if explore < .1:
            #    explore = .25
            
    env.monitor.close()