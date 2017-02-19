from extensions import Extension

class DataStreamTrack(Extension):
    def __init__(self, data_stream, variables, prefix='', best_method=min, **kwargs):
        kwargs.setdefault('before_training', True)
        kwargs.setdefault('after_epoch', True)
        super(DataStreamTrack, self).__init__(**kwargs)
        self.variables = variables
        for i, ent in enumerate(self.variables):
            if type(ent) is not tuple:
                self.variables[i] = (ent, 'mean')
        self.prefix = prefix
        self.data_stream = data_stream
        self.do_all = True
        if best_method == min:
            self.best_method = [min]*len(variables)
        else:
            self.best_method = best_method
        self.bests = [0]*len(variables)
        for i,m in enumerate(self.best_method):
            if m == max:
                self.bests[i] = -100000
            elif m == min:
                self.bests[i] = 100000

    def get_name(self, name):
        return self.prefix + '_' + name if self.prefix != '' else name

    def do(self, method, *args):
        log = self.main_loop.log
        outputs = [0]*len(self.main_loop.outputs)
        after_outputs = {}
        count = 0
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            count += 1
            self.ordered_batch = [(batch[v.name] if v.name in batch else self.main_loop.static_input[v.name])
                                  for v in self.main_loop.model.inputs]
            ops = self.main_loop.test_algorithm(*self.ordered_batch)

            for i in range(len(self.variables)):
                v = self.variables[i][0]
                t = self.variables[i][1]
                if self.variables[i][1] == 'after':
                    continue

                val = ops[self.main_loop.output_map[v]]
                if t in ['mean','sum']:
                    outputs[self.main_loop.output_map[v]] += val
                elif t == 'last':
                    outputs[self.main_loop.output_map[v]] = val
            if self.main_loop.testing:
                break
            if not self.do_all:
                break
        for i,var in enumerate(self.variables):
            v = self.variables[i][0]
            t = self.variables[i][1]
            if t == 'mean':
                outputs[self.main_loop.output_map[v]] /= count
                if hasattr(outputs[self.main_loop.output_map[v]], 'shape'):
                    outputs[self.main_loop.output_map[v]] = \
                                      outputs[self.main_loop.output_map[v]].sum()
            if t == 'after':
                after_outputs[v] = self.variables[i][4](outputs[self.main_loop.output_map[self.variables[i][2]]], outputs[self.main_loop.output_map[self.variables[i][3]]])
                if self.best_method[i](after_outputs[v], self.bests[i]) != self.bests[i]:
                    self.bests[i] = after_outputs[v]
            elif self.best_method[i](outputs[self.main_loop.output_map[v]], self.bests[i]) != self.bests[i]:
                self.bests[i] = outputs[self.main_loop.output_map[v]]
            log.current_row[self.get_name(v)] = outputs[self.main_loop.output_map[v]] if t != 'after' else after_outputs[v]
            log.current_row['best_'+self.get_name(v)] = self.bests[i]
