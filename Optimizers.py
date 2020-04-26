from keras import optimizers

class OptimizerFactory:

    def sgd_optimizer(self, learning_rate, momentum):
        optimizer = optimizers.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=True)
        return optimizer

    def rmsprop_optimizer(self, learning_rate):
        optimizer = optimizers.rmsprop(learning_rate=learning_rate)
        return optimizer

    def adagrad_optimizer(self, learning_rate):
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
        return optimizer

    def adadelta_optimizer(self, learning_rate):
        optimizer = optimizers.Adadelta(learning_rate=1.0, rho=0.95)
        return optimizer

    def adamax_optimizer(self, learning_rate):
        optimizer = optimizers.adamax(lr=learning_rate)
        return optimizer

    def adam_optimizer(self, learning_rate):
        optimizer = optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
        return optimizer

    def nadam_optimizer(self, learning_rate):
        optimizer = optimizers.Nadam(learning_rate=learning_rate)
        return optimizer