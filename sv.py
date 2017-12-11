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
from sklearn.utils import shuffle
import bob.learn.em
import bob.bio.gmm.algorithm.IVector as IVector
from Helper import Helper

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
        frac = 1/num
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

def enroll_features(model, features, number_of_test_cases = 1):
    # Enroll (8 - number_of_test_cases) test features for each test speaker. 

    clients = []
    for i in range(len(features)):
        logging.info('enroll_features %d/%d' % (i + 1, len(features)))
        print('enroll_features %d/%d' % (i + 1, len(features)))
        temp_features = []
        for j in range(len(features[i]) - number_of_test_cases):
            temp_features.append(project_features(model, features[i][j]))
        average_ivector = numpy.mean(numpy.vstack(temp_features), axis = 0)
        if model.use_plda:
            client = bob.learn.em.PLDAMachine(model.plda_base)
            model.plda_trainer.enroll(client, average_ivector.reshape(1,-1))
            clients.append(client)
        else:
            clients.append(average_ivector)
    return clients

def save_enrolled_speakers(path, clients):
    # Save enrolled PLDA models to path.

    Helper.create_folder(path)
    for i, client in enumerate(clients):
        with bob.io.base.HDF5File(path + 'client-' + str(i) + '.hdf5', 'w') as f:
            client.save(f)
    print('clients saved!')
    return


def read_enrolled_speakers(path, iv, num):
    # Read num PLDA models from path.

    clients = []
    for i in range(num):
        with bob.io.base.HDF5File(path + 'client-' + str(i) + '.hdf5', 'r') as f:
            client = bob.learn.em.PLDAMachine(iv.plda_base)
            client.load(f)
            clients.append(client)
    return clients

def test_features_(model, clients, features, number_of_test_cases = 1):
    # Test (number_of_test_cases) features for each test speaker.

    scores, labels = [], []
    for i in range(len(clients)):
        logging.info('test_features %d/%d' % (i + 1, len(clients)))
        print('test_features %d/%d' % (i + 1, len(clients)))
        for j in range(len(features)):
            for k in range(len(features[j]) - number_of_test_cases, len(features[j])):
                score = model.score(clients[i], project_features(model, features[j][k]))
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
    
    # step 1. construct training and test file-lists
    print('step1...')
    train_filename = ['core.trn']
    test_filename = ['8conv.trn']
    train_files, test_files = construct_data(args.data_path, args.trn_path, train_filename, test_filename)

    # step 2. extract mfcc
    print('step2...')
    train_features = extract_features(train_files, len(train_files['filename']), args.frame_size, args.overlap_size)
    test_features =  extract_features(test_files, len(test_files['filename']), args.frame_size, args.overlap_size)
    Helper.save(train_features, 'train_core_all.pkl')
    Helper.save(test_features, 'test_features_all.pkl')
    #train_features = Helper.load('train_core_all.pkl')
    #test_features = Helper.load('test_features_all.pkl')
    
    # step 3. preprocess
    print('step3...')
    train_features, test_features = preproces_features(train_features, test_features) # normalize
    train_features = extra_prepro(train_features, args.train_sep_num) # separate training data

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
    iv.train_projector(train_features, output_folder_path + 'iv-all-plda-' + str(args.train_sep_num) + '.hdf5')
    #iv.load_projector(output_folder_path + 'iv-all-plda.hdf5')
    
    # step 5. enroll
    print('step5...')
    #clients = read_enrolled_speakers('clients/', iv, len(test_features))
    clients = enroll_features(iv, test_features, args.number_of_test_cases)
    #save_enrolled_speakers('clients/', clients)
    
    # step 6. test
    print('step6...')
    scores, labels = test_features_(iv, clients, test_features, args.number_of_test_cases)
    
    # step 7. summary
    print('step7...')
    p_scores, n_scores = scores[numpy.where(labels == True)], scores[numpy.where(labels == False)[0]]
    threshold = bob.measure.eer_threshold(n_scores, p_scores)
    far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
    Helper.output(output_folder_path + 'result.txt', 'threshold = %f, eer = %f%%, far = %f%%, frr = %f%%\n' % ( \
        threshold, max(far, frr) * 100, far * 100, frr * 100))
    Helper.generate_det_curve(p_scores, n_scores, output_folder_path + 'det_curve.png')

if __name__ == '__main__':
    main()
