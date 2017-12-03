# TensorFlow version of NIPS2016 soundnet


import os
os.environ["PYTHON_EGG_CACHE"] = "/tmp/python-eggs"
from util import load_from_txt
from model import Model
import tensorflow as tf
import numpy as np
import argparse
import sys

local_config = {  

            'batch_size': 1, 
            'eps': 1e-5,
            'sample_rate': 22050,
            'load_size': 22050*20,
            'name_scope': 'SoundNet',
            }

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Extract Feature')
    
    parser.add_argument('-t', '--txt', dest='audio_txt', help='target audio txt path. e.g., [demo.txt]', default='demo.txt')

    parser.add_argument('-o', '--outpath', dest='outpath', help='output feature path. e.g., [output]', default='output')

    parser.add_argument('-p', '--phase', dest='phase', help='demo or extract feature. e.g., [demo, extract]', default='extract')

    parser.add_argument('-m', '--layer', dest='layer_min', help='start from which feature layer. e.g., [1]', type=int, default=1)

    parser.add_argument('-x', dest='layer_max', help='end at which feature layer. e.g., [24]', type=int, default=None)
    
    parser.add_argument('-c', '--cuda', dest='cuda_device', help='which cuda device to use. e.g., [0]', default='0')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('-s', '--save', dest='is_save', help='Turn on save mode. [False(default), True]', action='store_true')
    parser.set_defaults(is_save=False)
    
    args = parser.parse_args()

    return args


def extract_feat(model, sound_input, config,file_names=''):
    #raw_file = open('./data/*')
    #lines = raw_file.readlines().split('\n')
    #print "lines:",lines                                           
    #print "let's see :",sound_input,np.shape(sound_input)
    layer_min = config.layer_min
    layer_max = config.layer_max if config.layer_max is not None else layer_min + 1
    
    # Extract feature
    features = {}
    feed_dict = {model.sound_input_placeholder: sound_input}

    for idx in xrange(layer_min, layer_max):
        feature = model.sess.run(model.layers[idx], feed_dict=feed_dict)
        features[idx] = feature
        if config.is_save:
            np.save(os.path.join(config.outpath, (file_names+'_tf_fea{}.npy').format(str(idx).zfill(2))), np.squeeze(feature))
            tmp = os.path.join(config.outpath,(file_names+'_tf_fea{}.npy').format(str(idx).zfill(2)))
            print "Save layer {} with shape {} as {}".format( \
                    idx, np.squeeze(feature).shape, tmp)
    
    return features


if __name__ == '__main__':

    args = parse_args()

    # Setup visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Load pre-trained model
    G_name = './models/sound8.npy'
    param_G = np.load(G_name).item()
        
    if args.phase == 'demo':
        # Demo
        sound_samples = [np.reshape(np.load('data/demo.npy'), [1, -1, 1, 1])]
    else: 
        # Extract Feature
        
        sound_samples,sample_names = load_from_txt(args.audio_txt, config=local_config)
       
    
    # Make path
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    # Init. Session
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    
    with tf.Session(config=sess_config) as session:
        # Build model
        model = Model(session, config=local_config, param_G=param_G)
        init = tf.global_variables_initializer()
        session.run(init)
        
        model.load()
    
        for i in range(len(sound_samples)):
        
	    #print "args.audio.txt",args.audio_txt
            output = extract_feat(model, sound_samples[i],args,sample_names[i])
