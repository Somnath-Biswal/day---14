'''a loss function measures how good our predictions are
we can use this to adjust the parameters of our network'''
from joelnet.tensor import Tensor
import numpy as np
class Loss:
    def loss(self,predicted:Tensor,actual:Tensor)-> float:
        raise NotImplementedError
# abstarct based loss class
    def grad(self,predicted:Tensor,actual:Tensor)->Tensor:
        raise NotImplementedError        
# grad gradient is the partial derivative of loss function wrt to other predicted things


class MSE(Loss):
    ''' MSE is mean square erroe , although we're just going to
    total sqaured error '''
    
    def loss(self,predicted:Tensor,actual:Tensor)-> float:
        return np.sum((predicted - actual)**2)

    def grad(self,predicted:Tensor,actual:Tensor)->Tensor:
        return 2*(predicted - actual)  
        # implementing an actual loss function