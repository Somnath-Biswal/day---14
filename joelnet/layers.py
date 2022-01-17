'''our neural nets made of layers 
each layers would need to pass its inputs foeward
and propagate its gradients backwards. for example a neural net would looklike


inputs -> Linear -> Tanh -> Lineaar => output '''
from _typeshed import Self
from typing import Dict,Callable

from joelnet.tensor import Tensor
import numpy as np
from numpy.random.mtrand import randn, random

class Layer:# as it is base class raise not implemented error
    def __init__(self) -> None:
        self.params:Dict[str,Tensor]={}# ineed a constructor
        self.grads:Dict[str,Tensor]={}# we find the inputs and outputs
    def forward(self , inputs: Tensor) -> Tensor:
        '''produce outputs corresponding to these inputs'''# forward would take some inputs which need a tensor and output a tensor
        raise NotImplementedError

    def backward(self,grad:Tensor)->Tensor:
        '''backpropagate this gradient through layer'''# backward takes gradient and returns a tensor
        raise NotImplementedError

class Linear(Layer):
    '''computes output = inputs@w + b matrix multiplied by some weight plus bias'''
    def __init__(self , input_size: int , output_size:int) -> None:# check if a string if a string here  string gets angy
        # tell the constructor the input size and the output size and my file returns all the constructors return None 
        # inputs will be  (batch_size,input_size)
        # outputs will be (batch_Size,output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size,output_size)        # we intializ eweights with random values
        self.params["b"] = np.random.randn(output_size)# broadcast correctly and it wiil be fine
        # tried to assign this to dictionary and this dictionary is not defined in the base class
        # now we have yo call the super class constructor to get the dict intialized

    # now we have to get the forward part where we push the inputs through forward layer
    def forward(self,inputs:Tensor)->Tensor:
        '''
        outputs = inputs@w+b
        '''
        self.inputs = inputs
        # save a copy of the inputs and that when will do the backprop 
        return inputs@self.params["w"]+ self.params["b"]
    def backward(self, grad: Tensor) -> Tensor:
        return super().backward(grad)
        ''' 
        y = f(x) and x = a*b +c
        then dy/da =  f'(x)*b
        and dy/db = f'(x)*a
        and dy/dc = f'(x)
        if y = f(x) and x = a@b +c
        then dy/da = f'(x)@b.T
        and dy/db = a.T@f'(x)
        and dy/dc = f'(x)
        '''
        self.grads["b"] = np.sum(grad,axis=0)# the outputs are in the batch dimension the gradients are output size by  batch size we dont want to explicity mention output size
        # so the outputs should be batch size 
        self.grads["w"] = self.inputs.T@grad
        return grad@self.params["w"].T
# we dont have grads defined either 
F = Callable[[Tensor],Tensor] # List of inputs and single output

class Activation(Layer):
    '''
    An activation layer just applies a function elementwise to its inputs
    '''
    def __init__(self, f:F, f_prime:F ) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime# takes a tensor and returns a tensor

    def forward(self,inputs : Tensor)-> Tensor:
        self.inputs = inputs# we have to save the inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        '''
        if y = f(x) and x = g(z)
        then dy/dz = f'(x)*g'(z)

        '''
        return self.f_prime(self.inputs)*grad
        

def tanh(x:Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x:Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh , tanh_prime)
    




  


        


    
        




    