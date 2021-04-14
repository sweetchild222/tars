def gradient_sgd():
    return {'type':'sgd', 'parameter':{'lr':0.01}}

def gradient_adam():
    return {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95, 'exp':1e-7}}

def gradient_RMSprop():
    return {'type':'rmsProp', 'parameter':{'lr':0.001, 'beta':0.95, 'exp':1e-8}}



def activation_linear():
    return {'type':'linear'}

def activation_relu():
    return {'type':'relu'}

def activation_leakyRelu():
    return {'type':'leakyRelu', 'parameter':{'alpha':0.0001}}

def activation_sigmoid():
    return {'type':'sigmoid'}

def activation_elu():
    return {'type':'elu', 'parameter':{'alpha':0.0001}}

def activation_tanh():
    return {'type':'tanh'}




def weight_init_glorot_normal():
    return {'type':'glorot', 'random':'normal'}

def weight_init_glorot_uniform():
    return {'type':'glorot', 'random':'uniform'}

def weight_init_he_normal():
    return {'type':'he', 'random':'normal'}

def weight_init_he_uniform():
    return {'type':'he', 'random':'uniform'}

def weight_init_lecun_normal():
    return {'type':'lecun', 'random':'normal'}

def weight_init_lecun_uniform():
    return {'type':'lecun', 'random':'uniform'}



def loss_categorical():
    return {'type':'categorical'}

def loss_binary():
    return {'type':'binary'}

def loss_meansquare():
    return {'type':'meansquare'}



def template_lstm(activation, weightInit, input_shape, classes, unroll, stateful):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'lstm', 'parameter':{'units':256, 'activation':activation, 'recurrent_activation':activation_sigmoid(), 'weight_init':weightInit, 'unroll':unroll, 'stateful':stateful}},
        {'type':'dense', 'parameter':{'units':128, 'activation':activation, 'weight_init':weightInit}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_linear(), 'weight_init':weightInit}}]

    return layers


def template_basic(activation, weightInit, input_shape, classes, unroll, stateful):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'basicRNN', 'parameter':{'units':256, 'activation':activation, 'weight_init':weightInit, 'unroll':unroll, 'stateful':stateful}},
        {'type':'dense', 'parameter':{'units':64, 'activation':activation, 'weight_init':weightInit}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':activation_linear(), 'weight_init':weightInit}}]

    return layers


def createLayersTemplate(modelType, activationType, weightInitType, input_shape, classes, unroll, statefull):

    modelTypeList = {'basic':template_basic, 'lstm':template_lstm}
    activationTypeList = {'elu':activation_elu, 'relu':activation_relu, 'leakyRelu':activation_leakyRelu, 'sigmoid':activation_sigmoid, 'tanh':activation_tanh, 'linear':activation_linear}
    weightInitTypeList = {'glorot_normal':weight_init_glorot_normal, 'glorot_uniform':weight_init_glorot_uniform, 'he_normal':weight_init_he_normal, 'he_uniform':weight_init_he_uniform, 'lecun_normal':weight_init_lecun_normal, 'lecun_uniform':weight_init_lecun_uniform}

    template = modelTypeList[modelType]
    activation = activationTypeList[activationType]
    weightInit = weightInitTypeList[weightInitType]

    return template(activation(), weightInit(), input_shape, classes, unroll, statefull)


def createGradientTemplate(gradientType):

    gradientTypeList = {'adam':gradient_adam, 'sgd':gradient_sgd, 'rmsProp':gradient_RMSprop}

    gradient = gradientTypeList[gradientType]

    return gradient()


def createLossTemplate(lossType):

    lossTypeList = {'categorical':loss_categorical, 'binary':loss_binary, 'meansquare':loss_meansquare}

    loss = lossTypeList[lossType]

    return loss()
