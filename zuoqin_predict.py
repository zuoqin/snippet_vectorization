import traceback

from common import common
from extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_TRACK_LENGTH = 8
MAX_TRACK_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
import tempfile

class ZuoqinPredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.track_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_track_length=MAX_TRACK_LENGTH,
                                        max_track_width=MAX_TRACK_WIDTH)

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self, data=None):
        input_filename = 'Input.java'
        if data is not None:
            f = open(input_filename, 'wb+')
            f.write(data)
            f.close()
            print(input_filename)

        name_probs = []
        notices = []
        vector = ''
        if 1==1:
            try:
                predict_lines, hash_to_string_dict = self.track_extractor.extract_paths(input_filename)
                print(predict_lines)
            except ValueError as e:
                print(e)
                return {'notices': notices, 'names': name_probs}
            raw_forecast_results = self.model.predict(predict_lines)
            method_forecast_results = common.parse_forecast_results(
                raw_forecast_results, hash_to_string_dict,
                self.model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)
            for raw_forecast, method_forecast in zip(raw_forecast_results, method_forecast_results):
                print('Original name:\t' + method_forecast.original_name)
                for name_prob_pair in method_forecast.forecasts:
                    print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))
                    name_probs.append({'name': name_prob_pair['name'], 'probability': name_prob_pair['probability']})
                print('notice:')
                for notice_obj in method_forecast.notice_paths:
                    print('%f\tcontext: %s,%s,%s' % (
                    notice_obj['score'], notice_obj['token1'], notice_obj['path'], notice_obj['token2']))
                    notices.append({'token1': notice_obj['token1'], 'token2': notice_obj['token2'], 'score': notice_obj['score'], 'path': notice_obj['path']})
                if 1==1 or self.config.EXPORT_BODY_VECTORS:
                    print('Code vector:')
                    print(' '.join(map(str, raw_forecast.code_vector)))
                    vector = ' '.join(map(str, raw_forecast.code_vector))
        #fo.close()
        return {'notices': notices, 'names': name_probs}
