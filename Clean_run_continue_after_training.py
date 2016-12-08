# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random
import nplot

# HYPERPARMETERS
H = 100
H2 = 30
batch_number = 50
gamma = 0.99
explore = 1
num_of_episodes_between_q_copies = 50
learning_rate=1e-3

    
    
if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    print "Gym input is ", env.action_space
    print "Gym observation is ", env.observation_space
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    
    tf.reset_default_graph()
    f = open('recording.csv', 'w')

    #First Q Network
    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], .1, 1.0))
    bias1 = tf.Variable(tf.random_uniform([H], .1, 1.0))
    
    w2 = tf.Variable(tf.random_uniform([H,H2], .1, 1.0))
    bias2 = tf.Variable(tf.random_uniform([H2], .1, 1.0))
    
    w3 = tf.Variable(tf.random_uniform([H2,env.action_space.n], .1, 1.0))
    bias3 = tf.Variable(tf.random_uniform([env.action_space.n], .1, 1.0))
    
    states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="states")  # This is the list of matrixes that hold all observations
    #actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="actions")
    
    hidden_1 = tf.nn.relu(tf.matmul(states, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    action_values = tf.matmul(hidden_2, w3) + bias3
    
    actions = tf.placeholder(tf.int32, [None], name="training_mask")
    one_hot_actions = tf.one_hot(actions, env.action_space.n)
    Q = tf.reduce_sum(tf.mul(action_values, one_hot_actions), reduction_indices=1) 

    w1_prime = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], .1, 1.0))
    bias1_prime = tf.Variable(tf.random_uniform([H], .1, 1.0))
    
    w2_prime = tf.Variable(tf.random_uniform([H,H2], .1, 1.0))
    bias2_prime = tf.Variable(tf.random_uniform([H2], .1, 1.0))

    
    w3_prime = tf.Variable(tf.random_uniform([H2,env.action_space.n], .1, 1.0))
    bias3_prime = tf.Variable(tf.random_uniform([env.action_space.n], .1, 1.0))
    
    #Second Q network
    
    next_states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="n_s") # This is the list of matrixes that hold all observations
    hidden_1_prime = tf.nn.relu(tf.matmul(next_states, w1_prime) + bias1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    next_action_values =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime
    #next_values = tf.reduce_max(next_action_values, reduction_indices=1)   
    
     #need to run these to assign weights from Q to Q_prime
    w1_prime_update= w1_prime.assign(w1)
    bias1_prime_update= bias1_prime.assign(bias1)
    w2_prime_update= w2_prime.assign(w2)
    bias2_prime_update= bias2_prime.assign(bias2)
    w3_prime_update= w3_prime.assign(w3)
    bias3_prime_update= bias3_prime.assign(bias3)
    
    #we need to train Q
    rewards = tf.placeholder(tf.float32, [None, ], name="rewards") # This holds all the rewards that are real/enhanced with Qprime
    #loss = (tf.reduce_mean(rewards - tf.reduce_mean(action_values, reduction_indices=1))) * one_hot
    loss = tf.reduce_mean(tf.square(rewards - Q)) #* one_hot  
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment
    
    max_episodes = 5000
    max_steps = 199

    D = []
    explore = .01 # fixed explore while using saved variables
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "/home/jonathan/Desktop/CONFIGURED/DeepQ-CartPole-v0/model.ckpt")
        
        #Copy Q over to Q_prime
        sess.run(w1_prime_update)
        sess.run(bias1_prime_update)
        sess.run(w2_prime_update)
        sess.run(bias2_prime_update)
        sess.run(w3_prime_update)
        sess.run(bias3_prime_update)
    
        for episode in xrange(max_episodes):
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
            f.write((str(reward_sum)+"\n"))
            reward_sum = 0
            new_state = env.reset()
            
            for step in xrange(max_steps):
                
                if (((episode + 49) % batch_number) == 0):
                    env.render()
                
                state = list(new_state);
                
                if explore > random.random():
                    action = env.action_space.sample()
                else:
                    #get action from policy
                    results = sess.run(action_values, feed_dict={states: np.array([new_state])})
                    #print results
                    action = (np.argmax(results[0]))

                    
                curr_action = action;
                
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward
                
                
                
                D.append([state, curr_action, reward, new_state, done])
                
                
                if len(D) > 500000:  #2500 episodes worth... never forgets anything
                    D.pop(0)
                #Training a Batch
                #samples = D.sample(50)
                sample_size = len(D)
                database_length = len(D)
                if sample_size > 50:   #processes 2.5 episodes worth each step... am I overtraining?
                    sample_size = 50
                else:
                    sample_size = sample_size
                 
                if database_length > 5000:  # Don't start training right away, would overfit those values
                    samples = [ D[i] for i in random.sample(xrange(len(D)), sample_size) ]
                    #print samples
                    new_states_for_q = [ x[3] for x in samples]
                    all_q_prime = sess.run(next_action_values, feed_dict={next_states: new_states_for_q})
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        #print i_sample
                        if i_sample[4] == True:
                            #print i_sample[2]
                            y_.append(i_sample[2])
                            #print y_
                        else:
                            this_q_prime = all_q_prime[ind]
                            maxq = max(this_q_prime)
                            #print maxq
                            y_.append(i_sample[2] + (gamma * maxq))
                            #print y_
                        #y_.append(i_sample[2])
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(i_sample[1])
                    
                    #print sess.run(loss, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples})#feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples, one_hot: actions_samples})
                    sess.run(train, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples})
                        #y_ = reward + gamma * sess.run(next_action_values, feed_dict={next_states: np.array([i_sample[3]])})
                    #y_ = curr_action * np.vstack([y_])
                    #print y_
                    #y_ = y_
                    #print y_
                    #sess.run(train, feed_dict={states: np.array([i_sample[0]]), next_states: np.array([i_sample[3]]), rewards: y_, actions: np.array([i_sample[1]]), one_hot: np.array([curr_action])})
                    
                    if done:
                        break
                        
            if episode % num_of_episodes_between_q_copies == 0:
                sess.run(w1_prime_update)
                sess.run(bias1_prime_update)
                sess.run(w2_prime_update)
                sess.run(bias2_prime_update)
                sess.run(w3_prime_update)
                sess.run(bias3_prime_update)
            
            #explore = explore * .999   
 
            if episode % 500 == 0:
                save_path = saver.save(sess, "/home/jonathan/Desktop/CONFIGURED/DeepQ-CartPole-v0/model.ckpt")
                print("Model saved in file: %s" % save_path)
    env.monitor.close()
