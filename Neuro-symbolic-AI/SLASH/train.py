from data_generation import Intellizenz

program ='''
tab(t1).
pred(p1).

npp(vgsegment(1,T),[0,1,2]) :- tab(T).
event(N,C) :- vgsegment(0,+T,-C), pred(N).

'''

# Query
#:- not event(p1,1).