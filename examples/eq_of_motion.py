#imports
from __future__ import division, print_function
import sympy as sm
from sympy import trigsimp
import sympy.physics.mechanics as me
import mpmath as mp

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

# Defining symbolic Variables

n = 1
q = me.dynamicsymbols('q:{}'.format(n + 1))  # generalized coordinates
qdot = me.dynamicsymbols('qdot:{}'.format(n + 1))  #generalized speeds
f = me.dynamicsymbols('f')
m = sm.symbols('m:{}'.format(n + 1))
J = sm.symbols('J:{}'.format(n + 1))
l = sm.symbols('l:{}'.format(n))  # lenght of each pendlum
a = sm.symbols('a:{}'.format(n))  #location of Mass-centers
d = sm.symbols('d1:{}'.format(n + 1))  #viscous damping coef.
g, t = sm.symbols('g t')

# intertial reference frame
In_frame = me.ReferenceFrame('In_frame')

# Origninal Point O in Reference Frame :
O = me.Point('O')
O.set_vel(In_frame, 0)

# The Center of Mass Point on cart :
C0 = me.Point('C0')
C0.set_pos(O, q[0] * In_frame.x)
C0.set_vel(In_frame, qdot[0] * In_frame.x)

cart_inertia_dyadic = me.inertia(In_frame, 0, 0, J[0])
cart_central_intertia = (cart_inertia_dyadic, C0)

cart = me.RigidBody('Cart', C0, In_frame, m[0], cart_central_intertia)

kindiffs = [q[0].diff(t) - qdot[0]]  # entforcing qdot=Omega

frames = [In_frame]
mass_centers = [C0]
joint_centers = [C0]
central_intertias = [cart_central_intertia]

forces = [(C0, f * In_frame.x - m[0] * g * In_frame.y)]

#adding torques :
if n == 1:
    torqueVector = (-d[0] * qdot[1]) * In_frame.z
elif n >= 2:
    torqueVector = (-d[0] * qdot[1] - d[1] * (qdot[1] - qdot[2])) * In_frame.z

torques = [(In_frame, torqueVector)]

# cart_potential = 1 / 2 * d[0] * qdot[1]**2
# potentials = [cart_potential]
# cart.potential_energy= cart_potential


rigid_bodies = [cart]
# Lagrangian0 = me.Lagrangian(In_frame, rigid_bodies[0])
# Lagrangians=[Lagrangian0]


for i in range(n):
    #Creating new reference frame
    Li = In_frame.orientnew('L' + str(i), 'Axis', [q[i + 1], In_frame.z])
    Li.set_ang_vel(In_frame, qdot[i + 1] * In_frame.z)
    frames.append(Li)

    # Creating new mass point centers
    Pi = mass_centers[-1].locatenew('a' + str(i + 1), a[i] * Li.x)
    Pi.v2pt_theory(joint_centers[-1], In_frame, Li)
    mass_centers.append(Pi)

    #Creating new joint Points
    Jointi = joint_centers[-1].locatenew('jont' + str(i + 1), l[i] * Li.x)
    Jointi.v2pt_theory(joint_centers[-1], In_frame, Li)
    joint_centers.append(Jointi)

    #adding forces
    forces.append((Pi, -m[i + 1] * g * In_frame.y))
    
    
    #adding torqes
    if i==0 :
        torqueVectori= -d[0] * qdot[1] * In_frame.z
        torques.append((Li, torqueVectori))
    else:
        torqueVectori = -d[i] * (qdot[i+1]-qdot[i])*In_frame.z
        torques.append((Li, torqueVectori))
    
    #adding cential inertias 
    IDi = me.inertia(frames[i + 1], 0, 0, J[i + 1])
    ICi = (IDi, mass_centers[i + 1])
    central_intertias.append(ICi)

    LBodyi = me.RigidBody('L' + str(i + 1) + '_Body', mass_centers[i + 1],
                          frames[i + 1], m[i + 1], central_intertias[i + 1])
    rigid_bodies.append(LBodyi)

    kindiffs.append(q[i + 1].diff(t) - qdot[i + 1])
    """
    #potentials for Lagrangian
    h = mass_centers[i + 1].pos_from(O).dot(
        In_frame.y)  # height of the mass-center from  Point O

    if i <= n-2 :
        if i == n-2 :
            Potentiali= 1/2 * d[i+1]*(qdot[i+1])**2 + m[i+1]* g * h   
        else:
            Potentiali= 1/2 * d[i+1]*(qdot[i+1]-qdot[i+2])**2 + m[i+1]* g * h 

        potentials.append(Potentiali)
        rigid_bodies[i + 1].potential_energy = potentials[i + 1]

    #Lagraniag for rigid Bodies 
    Lagrangiani = me.Lagrangian(In_frame, rigid_bodies[i+1])
    Lagrangians.append(Lagrangiani)
    """    


    
#generalized force
loads = torques + forces

#Kane's Method --> Equation of motion
Kane = me.KanesMethod(In_frame, q, qdot, kindiffs)
fr, frstar = Kane.kanes_equations(rigid_bodies, loads)

mass_matrix = trigsimp(Kane.mass_matrix_full)
forcing_vector = trigsimp(Kane.forcing_full)

"""
#Lagrage Method for comparision of the results!
Lagrangian=0
for i in Lagrangians :
    Lagrangian+=Lagrangians[i]

Lag_mehtod=me.LagrangesMethod(Lagrangian, q, forces, rigid_bodies)    
"""
print(forcing_vector)    