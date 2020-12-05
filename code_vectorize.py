from vocabularies import VocabType
from config import Config
from zuoqin_predict import ZuoqinPredictor
from model_base import CodeVectorizeModelBase


def load_main_model(config: Config) -> CodeVectorizeModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import CodeVectorizeModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import CodeVectorizeModel
    return CodeVectorizeModel(config)


if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_main_model(config)
    config.log('Done creating code vectorize model')

    if config.is_training:
        model.train()
    if config.SAVE_W2V is not None:
        model.save_word2vec_format(config.SAVE_W2V, VocabType.Token)
        config.log('Origin word vectors saved in word2vec text format in: %s' % config.SAVE_W2V)
    if config.SAVE_T2V is not None:
        model.save_word2vec_format(config.SAVE_T2V, VocabType.Target)
        config.log('Target word vectors saved in word2vec text format in: %s' % config.SAVE_T2V)
    if (config.is_testing and not config.is_training) or config.RELEASE:
        eval_results = model.evaluate()
        if eval_results is not None:
            config.log(
                str(eval_results).replace('topk', 'top{}'.format(config.TOP_USED_CONSIDERED_DURING_PREDICTION)))
    if config.PREDICT:
        predictor = ZuoqinPredictor(config, model)
        predictor.predict()
    model.close_session()
