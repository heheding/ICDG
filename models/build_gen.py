from models import gap
from models import functions


def Generator(gen):
    if gen == 'gru':
        return gap.Feature()
    if gen == 'gru12':
        return gap.Feature(is_cat=False)
    if gen == 'cnn':
        return gap.CNN_SL_bn()
    if gen == 'change':
        return gap.CNN()
    if gen == 'lstm':
        return gap.BiLSTM()
    if gen == 'cnnlstm':
        return gap.CNNLSTM()
    
    return gap.Feature()


def Predictor():
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    #     return svhn2mnist.Predictor()
    
    return gap.Predictor()

def Discriminator(discri):
    if discri == 'AR':
        return gap.Discriminator_AR()
    elif discri == 'TRANs':
        return gap.Discriminator_ATT()
    elif discri == 'ATT':
        return gap.Att(dim=128)
    else:
        return gap.Domain_classifier()

