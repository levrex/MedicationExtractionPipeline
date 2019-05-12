# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:25:35 2018

@author: tdmaarseveen
"""

with open("H:/CTAKES/apache-ctakes-4.0.0-src/ctakes-lvg-res/target/classes/org/apache/ctakes/lvg/data/HSqlDb/lvg2008.data", "rb") as f:
    byte = f.read(10000)
    print(byte)
        