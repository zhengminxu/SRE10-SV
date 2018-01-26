#!/usr/local/bin python3
# -*- coding: utf-8 -*-

import os
import copy
import glob
import numpy
import random
import logging
import argparse
import bob.measure
import bob.bio.gmm
import bob.io.base
from sklearn import preprocessing
import bob.learn.em
import bob.bio.gmm.algorithm.IVector as IVector
from Helper import Helper
import tensorflow as tf
from model_autoencoder import Autoencoder

def construct_data(data_path, trn_path, train_file, test_file):
    # Construct training and test file-lists.

    train_data, test_data = {}, {}

    training_files = []
    for text in train_file:
        filename = trn_path + text
        with open(filename, 'r') as f:
            temp = f.read().split('\n')
            training_files += [line.split()[2] for line in temp if len(line) > 0]
    
    train_data['filename'] = [[data_path + f.split(':')[0][:-3] + 'wav'] for f in training_files]
    train_data['channel'] = [[int(f.split(':')[1] == 'B')] for f in training_files]
    
    test_files = []
    for text in test_file:    
        filename = trn_path + text
        with open(filename, 'r') as f:
            temp = f.read().split('\n')
            test_files += [line.split()[2:] for line in temp if len(line) > 0]

    test_data['filename'] = [[data_path + f.split(':')[0][:-3] + 'wav' for f in speaker] for speaker in test_files]
    test_data['channel'] = [[int(f.split(':')[1] == 'B') for f in speaker] for speaker in test_files]

    return train_data, test_data

def extract_features(data, num, frame_size, overlap_size):
    # Extract mfcc features from the first num of data.

    clients = []
    i = 0
    for s_list, c_list in zip(data['filename'][:num], data['channel'][:num]):
        logging.info('extract_features %d/%d' % (i+1, num))
        print('extract_features %d/%d' % (i+1, num))
        clients.append([])
        for f, c in zip(s_list, c_list):
            clients[i].append(Helper.get_mfcc(f, c, frame_size, overlap_size))
        i += 1
    return numpy.array(clients)

def preproces_features(train_features, test_features):
    # Normalize features.
    
    def normalize_features(clients, m, v):
        for i in range(len(clients)):
            logging.info('preproces_features %d/%d' % (i + 1, len(clients)))
            for j in range(len(clients[i])):
                for k in range(len(clients[i][j])):
                    clients[i][j][k] = (clients[i][j][k] - m) / v
        return clients
    features = numpy.vstack(feature for client in train_features for feature in client)
    scaler = preprocessing.StandardScaler().fit(features)
    train_features = normalize_features(train_features, scaler.mean_, scaler.var_)

    features = numpy.vstack(feature for client in test_features for feature in client)
    scaler = preprocessing.StandardScaler().fit(features)
    test_features = normalize_features(test_features, scaler.mean_, scaler.var_)
    return train_features, test_features

def extra_prepro(features, num):
    # Separate features into num parts. (only for training data!)
    
    assert num > 1
    new_feat = []
    for speaker in features:
        assert len(speaker) == 1
        frame_len = speaker[0].shape[0]
        start = 0
        new_speaker = []
        frac = 1.0/num
        for i in numpy.arange(frac,1+frac,frac):
            new_speaker.append(speaker[0][int(start*frame_len):int(i*frame_len)])
            start = i
        assert len(new_speaker) == num
        new_feat.append(new_speaker)

    return new_feat

def project_features(model, features):
    # Project mfcc to i-vectors.

    temp_features = []
    for i in range(len(features)):
        temp_features.append(features[i])
    return model.project(numpy.array(temp_features))

def get_train_iv(train_features, iv, save_name):
    train_iv = []
    for i, speaker in enumerate(train_features):
        print(i)
        speaker_iv = []
        for dialog in speaker:
            if dialog.shape[0] != 0:
                speaker_iv.append(project_features(iv, dialog))
            else:
                speaker_iv = []
                break
        if speaker_iv != []:
            train_iv.append(speaker_iv)

    train_iv = numpy.array(train_iv)
    
    try:
        train_iv.dump(save_name)
    except OverflowError:
        print('can\'t save train_iv!')
    else:
        print('train_iv saved!')
    return train_iv

def train_autoencoder(iv):
    auto = Autoencoder(alpha=0.001, beta=0.2, learning_rate=0.01)
    epochs = 200
    batch_size = 256
    saver = tf.train.Saver()

    train_iv, valid_iv = iv[:int(0.9*len(iv))], iv[int(0.9*len(iv)):]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            nb = 0
            for i in range(0, train_iv.shape[0], batch_size):
                batch = train_iv[i:i+batch_size]
                loss1, _ = sess.run([auto.loss, auto.opt], {auto.input_: batch})
                valid_loss = sess.run(auto.loss, {auto.input_: valid_iv})
                nb += 1
                print('epoch: %d, batch: %d, train loss: %f, valid loss: %f' % (e+1, nb, loss1, valid_loss))
                with open('loss1.txt', 'a') as f:
                    f.write('epoch: %d, batch: %d, train loss: %f, valid loss: %f\n' % (e+1, nb, loss1, valid_loss))
            saver.save(sess, 'ckpt/model', global_step = e)
    

def predict_icode(sess, model, features):
    icode = sess.run(model.encoded, {model.input_: features})
    #print(icode.shape)
    return icode


def enroll_features(model, model_dir, features, number_of_test_cases = 1):
    # Enroll (8 - number_of_test_cases) test features for each test speaker. 

    clients = []
    auto = Autoencoder(alpha=0.001, beta=0.2, learning_rate=0.01)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_dir)
    for i in range(len(features)):
        logging.info('enroll_features %d/%d' % (i + 1, len(features)))
        print('enroll_features %d/%d' % (i + 1, len(features)))
        temp_features = []
        for j in range(len(features[i]) - number_of_test_cases):
            temp_features.append(project_features(model, features[i][j]))
        average_ivector = numpy.mean(numpy.vstack(temp_features), axis = 0)
        average_ivector = numpy.reshape(average_ivector, (-1,1,100))
        
        i_code = predict_icode(sess, auto, average_ivector)
        clients.append(i_code)
    return clients # # of clients x 100

def save_enrolled_speakers(path, clients):
    # Save enrolled PLDA models to path.

    Helper.create_folder(path)
    for i, client in enumerate(clients):
        client.dump(path+'client-'+str(i)+'.npy')
    print('clients saved!')
    return


def read_enrolled_speakers(path, iv, num):
    # Read num PLDA models from path.

    clients = []
    for i in range(num):
        client = numpy.load(path+'client-'+str(i)+'.npy')
        clients.append(client)
    return clients


def cosine_score(model, probe):
    model = numpy.reshape(model, (75,))
    probe = numpy.reshape(probe, (75,))
    return numpy.dot(model/numpy.linalg.norm(model), probe/numpy.linalg.norm(probe))

def test_features_(model, model_dir, clients, features, number_of_test_cases = 1):
    # Test (number_of_test_cases) features for each test speaker.

    scores, labels = [], []
    auto = Autoencoder(alpha=0.001, beta=0.2, learning_rate=0.01)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_dir)
    for i in range(len(clients)):
        logging.info('test_features %d/%d' % (i + 1, len(clients)))
        print('test_features %d/%d' % (i + 1, len(clients)))
        for j in range(len(features)):
            for k in range(len(features[j]) - number_of_test_cases, len(features[j])):
                i_vector = project_features(model, features[j][k])
                i_code = predict_icode(sess, auto, numpy.reshape(i_vector, (-1,1,100)))
                score = cosine_score(clients[i], i_code)
                scores.append(score)
                labels.append(i == j)
    return numpy.array(scores), numpy.array(labels)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_test_cases', default = 2, type = int)
    parser.add_argument('--frame_size', default = 512, type = int)
    parser.add_argument('--overlap_size', default = 256, type = int)
    parser.add_argument('--train_sep_num', default = 4, type = int)
    parser.add_argument('--output_folder_path', default = '')
    args = parser.parse_args()
    args.data_path = '/NVME_SSD/CHT/SRE10/sp10-01/data/'
    args.trn_path = '/NVME_SSD/CHT/SRE10/sp10-01/train/'
    return args

def main():
    # step 0. initial
    args = get_args()
    output_folder_path = Helper.create_folder(args.output_folder_path)
    logging.basicConfig(
        level = logging.DEBUG, \
        filename = output_folder_path + 'log.txt', \
        format = '%(asctime)s - %(name)s[%(filename)s:%(lineno)d] - %(levelname)s - %(message)s' \
    )
    '''
    # step 1. construct training and test file-lists
    print('step1...')
    train_filename, test_filename = ['core.trn'], ['8conv.trn']
    train_files, test_files = construct_data(args.data_path, args.trn_path, train_filename, test_filename)

    # step 2. extract mfcc
    print('step2...')
    train_features = extract_features(train_files, len(train_files['filename']), args.frame_size, args.overlap_size)
    test_features =  extract_features(test_files, len(test_files['filename']), args.frame_size, args.overlap_size)
    Helper.save(train_features, 'train_core_all.pkl')
    Helper.save(test_features, 'test_features_all.pkl')
    '''
    train_features = Helper.load('train_core_all.pkl')
    test_features = Helper.load('test_features_all.pkl')
    
    # step 3. preprocess
    print('step3...')
    train_features, test_features = preproces_features(train_features, test_features) # normalize
    train_features = extra_prepro(train_features, args.train_sep_num) # separate training data
    # #of speakers x 4 x 256 x 39
    
    # step 4. training
    print('step4...')
    iv = IVector( \
        number_of_gaussians = 128, \
        gmm_training_iterations = 20, \
        use_whitening = True, \
        use_wccn = False, \
        use_lda = True, \
        lda_dim = 100, \
        use_pinv = True, \
        subspace_dimension_of_t = 300,
        tv_training_iterations = 5, \
        use_plda = True, \
        plda_dim_F  = 50, \
        plda_dim_G = 50, \
        plda_training_iterations= 50, \
    )
    #iv.train_projector(train_features, output_folder_path + 'iv-all-plda-' + str(args.train_sep_num) + '.hdf5')
    iv.load_projector(output_folder_path + 'iv-all-plda.hdf5')
    
    # step 4.5 train autoencoder
    #train_iv = get_train_iv(train_features, iv, 'train_iv_all.npy') # shape: (#of speakers x 4) x 100
    train_iv = numpy.load('train_iv_all.npy')
    train_autoencoder(train_iv)
    
    # step 5. enroll
    print('step5...')
    #clients = read_enrolled_speakers('clients-auto/', iv, len(test_features))
    model_dir = tf.train.latest_checkpoint('ckpt/')
    clients = enroll_features(iv, model_dir, test_features, args.number_of_test_cases)
    #save_enrolled_speakers('clients-auto/', clients)
    
    # step 6. test
    print('step6...')
    scores, labels = test_features_(iv, model_dir, clients, test_features, args.number_of_test_cases)
    
    # step 7. summary
    print('step7...')
    p_scores, n_scores = scores[numpy.where(labels == True)].astype(numpy.double), scores[numpy.where(labels == False)[0]].astype(numpy.double)
    threshold = bob.measure.eer_threshold(n_scores, p_scores)
    far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
    Helper.output(output_folder_path + 'result.txt', 'threshold = %f, eer = %f%%, far = %f%%, frr = %f%%\n' % ( \
        threshold, max(far, frr) * 100, far * 100, frr * 100))
    Helper.generate_det_curve(p_scores, n_scores, output_folder_path + 'det_curve.png')

if __name__ == '__main__':
    main()
