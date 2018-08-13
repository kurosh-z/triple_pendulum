# coding: utf-8
import sympy as sm
import sympy.physics.mechanics as me
import dill 

frame = me.ReferenceFrame('frame')
 
with open('test.pkl', 'wb') as file :
    dill.dump(frame, file)

with open('test.pkl', 'rb') as file :
    frame_=dill.load(file)

    assert frame_== frame