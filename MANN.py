import numpy as np
import tensorflow as tf
import pandas as pd
from mann import Gating as GT
from mann.Gating import Gating
from mann import ExpertWeights as EW
from mann.ExpertWeights import ExpertWeights
from mann.AdamWParameter import AdamWParameter
from mann.AdamW import AdamOptimizer
from mann import Utils as utils


class MANN(object):
    def __init__(self, 
                 input_size,
                 output_size,
                 rng,
                 sess:tf.Session,
                 datapath, savepath,
                 num_experts,
                 hidden_size = 512,
                 hidden_size_gt = 32, 
                 index_gating = [10, 15, 19, 23],
                 batch_size = 32 , epoch = 150, Te = 10, Tmult =2, 
                 learning_rate_ini = 0.0001, weightDecay_ini = 0.0025, keep_prob_ini = 0.7):
        
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rng         = rng
        self.sess        = sess
        
        shuffle_seed     = rng.randint(0xffff)
        #load data
        self.savepath    = savepath
        self.datapath    = datapath

        self.input_data  = tf.data.experimental.CsvDataset(datapath + '/Input.csv', header=True, record_defaults=[tf.float32]*input_size).shuffle(buffer_size=10000, seed=shuffle_seed).batch(batch_size, drop_remainder=True)
        # self.input_data  = tf.data.Dataset.from_tensor_slices(pd.read_csv(datapath + '/Input.csv')).shuffle(buffer_size=10000, seed=shuffle_seed).batch(batch_size, drop_remainder=True)
        self.input_mean  = np.fromfile(datapath+'/Input.mean.bin')
        self.input_std   = np.fromfile(datapath+'/Input.std.bin')

        self.output_data = tf.data.experimental.CsvDataset(datapath + '/Output.csv', header=True, record_defaults=[tf.float32]*output_size).shuffle(buffer_size=10000, seed=shuffle_seed).batch(batch_size, drop_remainder=True)
        # self.output_data = tf.data.Dataset.from_tensor_slices(pd.read_csv(datapath + '/Output.csv')).shuffle(buffer_size=10000, seed=shuffle_seed).batch(batch_size, drop_remainder=True)
        self.output_mean = np.fromfile(datapath+'/Output.mean.bin')
        self.output_std  = np.fromfile(datapath+'/Output.std.bin')
        
        #gatingNN
        self.num_experts    = num_experts
        self.hidden_size_gt = hidden_size_gt
        self.index_gating   = index_gating
        
        #training hyperpara
        self.batch_size    = batch_size
        self.epoch         = epoch

        # calcuclate total_batch
        total_batch = 0
        it = self.input_data.make_one_shot_iterator().get_next()
        self.sess.run(tf.global_variables_initializer())
        while True:
            try:
                self.sess.run([it])
                total_batch += 1
            except tf.errors.OutOfRangeError:
                break
    
        self.total_batch = total_batch
        print('total batches: %d' % total_batch)

        #adamWR controllers
        self.AP = AdamWParameter(nEpochs      = self.epoch,
                                 Te           = Te,
                                 Tmult        = Tmult,
                                 LR           = learning_rate_ini, 
                                 weightDecay  = weightDecay_ini,
                                 batchSize    = self.batch_size,
                                 nBatches     = self.total_batch
                                 )
        #keep_prob
        self.keep_prob_ini     = keep_prob_ini
        
        
        
        
    def build_model(self):
        #Placeholders
        self.nn_X         = tf.placeholder(tf.float32, [self.batch_size, self.input_size],  name='nn_X') 
        self.nn_Y         = tf.placeholder(tf.float32, [self.batch_size, self.output_size], name='nn_Y')  
        self.nn_keep_prob = tf.placeholder(tf.float32, name = 'nn_keep_prob') 
        self.nn_lr_c      = tf.placeholder(tf.float32, name = 'nn_lr_c') 
        self.nn_wd_c      = tf.placeholder(tf.float32, name = 'nn_wd_c')
        
        """BUILD gatingNN"""
        #input of gatingNN
        self.input_size_gt  = len(self.index_gating)
        self.gating_input   = tf.transpose(GT.getInput(self.nn_X, self.index_gating))
        self.gatingNN = Gating(self.rng, self.gating_input, self.input_size_gt, self.num_experts, self.hidden_size_gt, self.nn_keep_prob)
        #bleding coefficients
        self.BC = self.gatingNN.BC
        
        #initialize experts
        self.layer0 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size,  self.input_size),   'layer0') # alpha: 4/8*hid*in, beta: 4/8*hid*1
        self.layer1 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size, self.hidden_size),   'layer1') # alpha: 4/8*hid*hid,beta: 4/8*hid*1
        self.layer2 = ExpertWeights(self.rng, (self.num_experts, self.output_size, self.hidden_size),   'layer2') # alpha: 4/8*out*hid,beta: 4/8*out*1 
        
        
        #initialize parameters in main NN
        """
        dimension of w: ?* out* in
        dimension of b: ?* out* 1
        """
        w0  = self.layer0.get_NNweight(self.BC, self.batch_size)
        w1  = self.layer1.get_NNweight(self.BC, self.batch_size)
        w2  = self.layer2.get_NNweight(self.BC, self.batch_size)
        
        b0  = self.layer0.get_NNbias(self.BC, self.batch_size)
        b1  = self.layer1.get_NNbias(self.BC, self.batch_size)
        b2  = self.layer2.get_NNbias(self.BC, self.batch_size)
        
        #build main NN
        H0 = tf.expand_dims(self.nn_X, -1)                     #?*in -> ?*in*1
        H0 = tf.nn.dropout(H0, keep_prob=self.nn_keep_prob)        
        
        H1 = tf.matmul(w0, H0) + b0                            #?*out*in mul ?*in*1 + ?*out*1 = ?*out*1
        H1 = tf.nn.elu(H1)             
        H1 = tf.nn.dropout(H1, keep_prob=self.nn_keep_prob) 
        
        H2 = tf.matmul(w1, H1) + b1
        H2 = tf.nn.elu(H2)             
        H2 = tf.nn.dropout(H2, keep_prob=self.nn_keep_prob) 
        
        H3 = tf.matmul(w2, H2) + b2
        self.H3 = tf.squeeze(H3, -1)                           #?*out*1 ->?*out  
        
        self.loss       = tf.reduce_mean(tf.square(self.nn_Y - self.H3))
        self.optimizer  = AdamOptimizer(learning_rate= self.nn_lr_c, wdc =self.nn_wd_c).minimize(self.loss)
        
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        """training"""
        #randomly select training set
        error_train = np.ones(self.epoch)
        #saving path
        model_path   = self.savepath+ '/model'
        nn_path      = self.savepath+ '/nn'
        weights_path = self.savepath+ '/weights'
        utils.build_path([model_path, nn_path, weights_path])
        
        #start to train
        print('Learning starts..')
        for epoch in range(self.epoch):
            avg_cost_train = 0
            it_input = tf.transpose(self.input_data.make_one_shot_iterator().get_next())
            it_output = tf.transpose(self.output_data.make_one_shot_iterator().get_next())
            # it_input = self.input_data.make_one_shot_iterator().get_next()
            # it_output = self.output_data.make_one_shot_iterator().get_next()
            for i in range(self.total_batch):
                batch_xs, batch_ys = self.sess.run([it_input, it_output])
                clr, wdc = self.AP.getParameter(epoch)   #currentLearningRate & weightDecayCurrent
                l, _, = self.sess.run([self.loss, self.optimizer], feed_dict={
                    self.nn_X: batch_xs,
                    self.nn_Y: batch_ys,
                    self.nn_keep_prob: self.keep_prob_ini,
                    self.nn_lr_c: clr,
                    self.nn_wd_c: wdc,
                })
                avg_cost_train += l / self.total_batch
                
                if i % 1000 == 0:
                    print(i, "trainingloss:", l)
                    print('Epoch:', '%04d' % (epoch + 1), 'clr:', clr)
                    print('Epoch:', '%04d' % (epoch + 1), 'wdc:', wdc)
                    
            #print and save training test error 
            print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
            
            error_train[epoch] = avg_cost_train
            error_train.tofile(model_path+"/error_train.bin")

            #save model and weights
            saver.save(self.sess, model_path+"/model.ckpt")
            GT.save_GT((self.sess.run(self.gatingNN.w0), self.sess.run(self.gatingNN.w1), self.sess.run(self.gatingNN.w2)), 
                       (self.sess.run(self.gatingNN.b0), self.sess.run(self.gatingNN.b1), self.sess.run(self.gatingNN.b2)), 
                       nn_path
                       )
            EW.save_EP((self.sess.run(self.layer0.alpha), self.sess.run(self.layer1.alpha), self.sess.run(self.layer2.alpha)),
                       (self.sess.run(self.layer0.beta), self.sess.run(self.layer1.beta), self.sess.run(self.layer2.beta)),
                       nn_path,
                       self.num_experts
                       )
            
            if epoch%10==0:
                weights_nn_path = weights_path + '/nn%03i' % epoch
                utils.build_path([weights_nn_path])
                GT.save_GT((self.sess.run(self.gatingNN.w0), self.sess.run(self.gatingNN.w1), self.sess.run(self.gatingNN.w2)), 
                           (self.sess.run(self.gatingNN.b0), self.sess.run(self.gatingNN.b1), self.sess.run(self.gatingNN.b2)), 
                           weights_nn_path
                           )
                EW.save_EP((self.sess.run(self.layer0.alpha), self.sess.run(self.layer1.alpha), self.sess.run(self.layer2.alpha)),
                           (self.sess.run(self.layer0.beta), self.sess.run(self.layer1.beta), self.sess.run(self.layer2.beta)),
                           weights_nn_path,
                           self.num_experts
                           )
        print('Learning Finished')

    def evaluate(self):
        X = tf.contrib.data.CsvDataset(self.datapath+'/Input.test.csv',  header=True, record_defaults=[tf.float32]*self.num_inputs ).batch(self.batch_size)
        Y = tf.contrib.data.CsvDataset(self.datapath+'/Output.test.csv', header=True, record_defaults=[tf.float32]*self.num_outputs).batch(self.batch_size)

        it_x = X.make_one_shot_iterator().get_next()
        it_y = Y.make_one_shot_iterator().get_next()
        loss = 0.0
        num_batch = 0
        while True:
            try:
                batch_x, batch_y = self.sess.run([it_x, it_y])
                feed_dict = {self.nn_X: batch_x, self.nn_Y: batch_y, self.nn_keep_prob: 1.0}
                [l] = self.sess.run([self.loss], feed_dict=feed_dict)
                loss += l
                num_batch += 1

            except tf.errors.OutOfRangeError:
                break

        print('evaluation loss: %f' % loss/num_batch)
