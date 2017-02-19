from extensions import Extension
import os
import cPickle

class SaveModel(Extension):

    def __init__(self, subdir, model_name, **kwargs):
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('after_training', True)
        super(SaveModel, self).__init__(**kwargs)
        self.subdir = subdir
        self.model_name = model_name

    def do(self, *args):
        f = file(os.path.join(self.subdir, self.model_name), 'wb')
        cPickle.dump(self.main_loop.model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


class SaveBestModel(SaveModel):
    
    def __init__(self, variable, **kwargs):
        super(SaveBestModel, self).__init__(**kwargs)
        self.variable = variable

    def do(self, *args):
        if self.main_loop.log.status['_best_epoch'+self.variable]:
            super(SaveBestModel, self).do(*args)
