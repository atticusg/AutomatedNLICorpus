from __future__ import absolute_import, division, print_function

import scipy
import sys, os, time, random, collections, itertools
import tensorflow as tf
import util
import numpy as np
from attentionmodel import PIModel
from configs.standard_conf import config

def main(_):
    #exec("from configs.{} import config".format(sys.argv[1]))

    ## get embedding
    word_to_id = util._get_word_to_id("glove/glove.6B.list", vocab_limit=config.vocab_limit)
    pretrained_embeddings = util._get_glove_vec("glove/glove.6B.300d.txt", vocab_limit=config.vocab_limit)

    ## Initialize model and tf session

    m = PIModel(config, pretrained_embeddings)
    labels = ['entails','contradicts','permits']
    cat_names = ['{}=>{}'.format(x,y) for x,y in itertools.product(labels,labels)]
    logs = []
    trains = []
    lrs = [0.0005, 0.001, 0.005]
    dropout = [1,0.9,0.8]
    l2 = [0.001, 0.0005, 0.0001]
    for b in l2:
        for d in dropout:
            for lr in lrs:
                config.l2 = b
                config.dropout = d
                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    ## Iterate
                    print("\n\n\n")
                    print(lr)
                    print(config.dropout)
                    print(config.l2)
                    for ep in range(2):#config.num_epoch):
                        print("Begin epoch {}".format(ep))
                        
                        ### Call training and validation routine
                        ### This part runs one epoch of training and one epoch of validation
                        ### Outcomes: preds, labels, constr, loss
                        train_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='train', prefix='pi', shuffle=False)
                        preds_t, labels_t, loss_t = m.run_train_epoch(sess, train_data, [lr])
                        val_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='val', prefix='pi', shuffle=False)
                        preds_v, labels_v, loss_v = m.run_val_epoch(sess, val_data)
                        test_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='test', prefix='pi', shuffle=False)
                        preds_v2, labels_v2, loss_v2 = m.run_val_epoch(sess, test_data)
                        jointtest= util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='jointtest', prefix='pi', shuffle=False)
                        preds_v3, labels_v3, loss_v3 = m.run_val_epoch(sess, jointtest)
                        
                        ### This part evaluates the model performance: predictions (preds) against labels (labels)
                        ###     metric: f1_macro, f1_micro, accuracy, confusion_matrix
                        ###     by_construct: Want results separately for each implicative construct?
                        ###     constr: (used when by_consruct is true) the source constructs for each prediction/label pairs
                        s0 = util.get_evaluation(preds_t, labels_t, metric='f1_macro')
                        s1 = util.get_evaluation(preds_t, labels_t, metric='f1_micro')
                        s2 = util.get_evaluation(preds_v, labels_v, metric='f1_macro')
                        s3 = util.get_evaluation(preds_v, labels_v, metric='f1_micro')
                        
                        s_cat = util.get_evaluation(preds_v, labels_v, metric='confusion_matrix')
                        
                        #for constr in s_cat:
                        #    logs[constr].append(s_cat[constr])  # write confusion matrix to log
                        
                        ### Print the performance
                        print("Train loss = {} :: Val loss = {}".format(loss_t, loss_v))
                        print("Train F1 macro/micro = {}/{} :: Val F1 macro/micro = {}/{}".format(s0, s1, s2, s3))
                        train_results = []
                        test_results = []
                        jointtest_results = []
                        mat = s_cat / np.sum(s_cat, axis=1, keepdims=True)
                        print("Train loss = {} :: Val loss = {} :: Test loss = {}".format(loss_t, loss_v, loss_v2))
                        right = 0
                        total = 0
                        c = 0
                        ct =0
                        e = 0
                        et = 0
                        p = 0
                        pt = 0
                        for x, y in zip(preds_t, labels_t):
                            train_results.append((x,y))
                            if y == 0:
                                et += 1
                            if y == 1:
                                ct += 1
                            if y == 2:
                                pt += 1
                            if x == y:
                                right += 1
                                if y == 0:
                                    e += 1
                                if y == 1:
                                    c += 1
                                if y == 2:
                                    p += 1
                            total += 1
                        print("train")
                        print(right/total)
                        print("entails", e/et)
                        print("contradicts", c/ct)
                        print("permits", p/pt)
                        right = 0
                        total = 0
                        c = 0
                        ct =0
                        e = 0
                        et = 0
                        p = 0
                        pt = 0
                        for x, y in zip(preds_v, labels_v):
                            if y == 0:
                                et += 1
                            if y == 1:
                                ct += 1
                            if y == 2:
                                pt += 1
                            if x == y:
                                right += 1
                                if y == 0:
                                    e += 1
                                if y == 1:
                                    c += 1
                                if y == 2:
                                    p += 1
                            total += 1
                        print("valid")
                        print(right/total)
                        print("entails", e/et)
                        print("contradicts", c/ct)
                        print("permits", p/pt)
                        right = 0
                        total = 0
                        c = 0
                        ct =0
                        e = 0
                        et = 0
                        p = 0
                        pt = 0
                        for x, y in zip(preds_v2, labels_v2):
                            test_results.append((x,y))
                            if y == 0:
                                et += 1
                            if y == 1:
                                ct += 1
                            if y == 2:
                                pt += 1
                            if x == y:
                                right += 1
                                if y == 0:
                                    e += 1
                                if y == 1:
                                    c += 1
                                if y == 2:
                                    p += 1
                            total += 1
                        print("test")
                        print(right/total)
                        print("entails", e/et)
                        print("contradicts", c/ct)
                        print("permits", p/pt)
                        right = 0
                        total = 0
                        c = 0
                        ct =0
                        e = 0
                        et = 0
                        p = 0
                        pt = 0
                        for x, y in zip(preds_v3, labels_v3):
                            jointtest_results.append((x,y))
                            if y == 0:
                                et += 1
                            if y == 1:
                                ct += 1
                            if y == 2:
                                pt += 1
                            if x == y:
                                right += 1
                                if y == 0:
                                    e += 1
                                if y == 1:
                                    c += 1
                                if y == 2:
                                    p += 1
                            total += 1
                        print("jointtest")
                        print(right/total)
                        print("entails", e/et)
                        print("contradicts", c/ct)
                        print("permits", p/pt)
                        if False:
                            with open("train"+ '.pkl', 'wb') as f:
                                pickle.dump(train_results, f, pickle.HIGHEST_PROTOCOL)
                            with open("test"+ '.pkl', 'wb') as f:
                                pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)
                            with open("jointtest"+ '.pkl', 'wb') as f:
                                pickle.dump(jointtest_results, f, pickle.HIGHEST_PROTOCOL)
                        #print(mat.reshape(9))
                        #logs.append(mat.reshape(9))
                        #trains.append("train" + str(loss_t) + "valid" + str(loss_v))
                        #print("End of epoch {}\n".format(ep))
                        if 1 == 0: 
                            time_train_done = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                            dest_dir = os.path.join("results", "{}-{}".format(sys.argv[1], time_train_done))
                            os.makedirs(dest_dir)    
                            print(tf.train.Saver().save(sess, os.path.join(dest_dir, 'model' + str(ep))))
                            with open(os.path.join(dest_dir,'log'), 'w') as f:
                                for cat_name, scores in zip(cat_names, zip(*logs)):
                                    f.write(cat_name + " " + " ".join(map(str, scores)) + "\n")
                                for train in trains:
                                    f.write(train + "\n")


if __name__ == '__main__':
    tf.app.run()
