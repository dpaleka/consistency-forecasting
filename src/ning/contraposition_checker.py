# -*- coding: utf-8 -*-
"""contraposition_checker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TEKwUq4nHlbQbDSEDBklWTZdMj18xpBx
"""

import negation_checker
from negation_checker import ask, neg

def combine(s1, s2):
  return("if "+s1+", "+"then "+s2)

def check_statements(statements, i):
  for _ in range(i):
     s1 = random_choice(statements)
     s2 = random_choice(statements)
     if !check(s1,s2):
       return false

  return true





def check(s1, s2):
  p = ask(combine(s1, s2), model)
  st1 = neg(s1)
  st2 = neg(s2)
  q = ask(combine(st2, st1), model)
  return (p==q)