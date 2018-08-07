# coding: utf-8


class Pen_Container(object):
    def __init__(self, label, **kwargs):
        assert len(set(dir(self)).intersection(list(kwargs.keys()))) == 0
        self.__dict__.update(kwargs)
        self.label = label

global pendata

def Pen_Container_initializer(number_of_pendulums):
    
    '''
    Create an Instance of Pen_container so that 
    there is just one Instance of it across the package !

    '''

    global pendata

    # defining general parameters
    parameter_values_simple_pendulum = [
        0.32,  # l[0] : l1
        0.2,  # a[0]
        3.34,  # m[0]
        0.8512,  # m[1]
        0.0,  # J[0]
        0.01980,  # J[1]
        0.010,  # d[0]    
        9.81,  # g
        0.0  # f 
    ]

    parameter_values_double_pendulum = [
        0.32,  # l[0] : l1
        0.419,  # l[1] : l2
        0.2,  # a[0]
        0.26,  # a[1]
        3.34,  # m[0]
        0.8512,  # m[1]
        0.8973,  # m[2]
        0.0,  # J[0]
        0.01980,  # J[1]
        0.02105,  # J[2]
        0.01980,  # d[0]
        1.9e-6,  # d[1]
        9.81,  # g
        0.0  # f 
    ]

    parameter_values_triple_pendulum = [
        0.32,  # l[0] : l1
        0.419,  # l[1] : l2
        0.485,  # l[2] : l3
        0.2,  # a[0] : a1
        0.26,  # a[1] : a2
        0.216,  # a[2] : a3
        3.34,  # m[0] : m0
        0.8512,  # m[1] : m1
        0.8973,  # m[2] : m2
        0.5519,  # m[3] : m3
        0.0,  # J[0] : J0
        0.01980,  # J[1] : J1
        0.02105,  # J[2] : J2
        0.01818,  # J[3] : J3
        0.01980,  # d[0] : d1
        1.9e-6,  # d[1] : d2
        0.00164,  # d[2] : d3
        9.81,  # g    : g
        0.0  # f    : f 
    ]

    parameter_values_dict = {
        'simple': parameter_values_simple_pendulum,
        'double': parameter_values_double_pendulum,
        'triple': parameter_values_triple_pendulum
    }
    
    if number_of_pendulums == 1 :
        label= 'simple'
    elif number_of_pendulums == 2 :
        label= 'double'
    elif number_of_pendulums == 3 :
        label = 'triple'

    # defining our data structure : 

    pendata = Pen_Container('Inverted' + '_' + label +'_' +'Pendulum')
    pendata.parameter_values= parameter_values_dict[label]
    pendata.model = Pen_Container('model')

    # trajectory :
    pendata.trajectory= Pen_Container('Trajecory')
    pendata.trajectory.k= None
    # pendata.trajectory.n=[]
    # pendata.trajectory.max_time
    # pendata.trajectory.fxu=lambda x, u: print('you have to define fxu !')
    
    # tracking
    pendata.tracking= Pen_Container('tracking')
    pendata.tracking.ucl=[]
    pendata.number_of_pendulums= number_of_pendulums
    
    print('somebody called me !')




'''

if __name__ == "__main__":
    Pen_Container_initializer()
    print('an instance of pen_container is produced')

'''