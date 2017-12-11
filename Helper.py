import os
import numpy
import dill
import scipy.io.wavfile
#import python_speech_features
import bob.bio.spear.preprocessor

class Helper:

    @staticmethod
    def output(filepath, content):
        with open(filepath, 'a') as f:
            f.write(content + '\n')

    @staticmethod
    def load(filepath):
        data = None
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data

    @staticmethod
    def save(variable, filepath):
        with open(filepath, 'wb') as f:
            dill.dump(variable, f)
        print('data saved successfully!')

    @staticmethod
    def create_folder(path):
        if len(path) > 0:
            path = path.rstrip('/') + '/'
            os.makedirs(path, exist_ok = True)
        return path

    @staticmethod
    def get_mfcc(file_path, channel, frame_size, overlap_size):
        (sample_rate, signal) =  scipy.io.wavfile.read(file_path)
        single_signal = signal[:,channel]
        vad = bob.bio.spear.preprocessor.Energy_Thr(
            smoothing_window = 10, \
            win_length_ms = frame_size / sample_rate * 1000, \
            win_shift_ms = (frame_size - overlap_size) / sample_rate * 1000, \
        )
        _, _, labels = vad([sample_rate, single_signal.astype('float')])
        if numpy.sum(labels) == 0:
            print('############# 0 ############')
            labels = numpy.ones_like(labels)
        extractor = bob.bio.spear.extractor.Cepstral(
        	n_ceps = 12, \
        	n_filters = 26, \
        	f_min = 0, \
        	f_max = sample_rate / 2, \
        	win_length_ms = frame_size / sample_rate * 1000, \
        	win_shift_ms = (frame_size - overlap_size) / sample_rate * 1000, \
        	pre_emphasis_coef = 0.97, \
        	features_mask = numpy.arange(0,39), # (12 + 1) * 3
        	normalize_flag = False,
        )
        mfcc = extractor([sample_rate,single_signal.astype('float'), labels])
        return mfcc

    @staticmethod
    def generate_det_curve(p_scores, n_scores, output_path):
        from matplotlib import pyplot
        pyplot.switch_backend('agg')
        bob.measure.plot.det(n_scores, p_scores, 1000, color = (0,0,0), linestyle = '-', label = 'test')
        bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
        pyplot.xlabel('FAR (%)')
        pyplot.ylabel('FRR (%)')
        pyplot.grid(True)
        pyplot.savefig(output_path)
        pyplot.cla()
        pyplot.clf()
