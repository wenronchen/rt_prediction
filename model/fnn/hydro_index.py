#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:04:12 2021

@author: wenrchen
"""

###different versions of hydrophobicity index


alphabet='ACDEFGHIKLMNPQRSTVWY'


###a.1982
hydro_index_a=[0,1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]



###b.1996
'''
#  "Experimentally determined hydrophobicity scale for proteins at
#  membrane interfaces," Wimley and White, Nat Struct Biol 3:842 (1996).
#
#  More positive means more hydrophobic.
attribute: wwHydrophobicity
'''
hydro_index_b=[0,-0.17,0.24,-1.23,-2.02,1.13,-0.01,-0.96,0.31,-0.99,0.56,0.23,-0.42,-0.45,-0.58,-0.81,-0.13,-0.14,\
             -0.07,1.85,0.94]


###c.2005
'''
#  the supplementary information for "Recognition of transmembrane 
#  helices by the endoplasmic reticulum translocon," Hessa et al., 
#  Nature 433:377 (2005).
#
#  More negative means more hydrophobic. 
attribute: hhHydrophobicity
'''
hydro_index_c=[0,0.11,-0.13,3.49,2.68,-0.32,0.74,2.06,-0.60,2.71,-0.55,-0.10,2.05,2.23,2.36,2.58,0.84,0.52,\
                      -0.31,0.30,0.68]

###d.2011
'''
#  Amino acid hydrophobicity scale from Table S1 (delta delta G) in
#  the supplementary information for "Side-chain hydrophobicity scale 
#  derived from transmembrane protein folding into lipid bilayers."
#  Moon CP, Fleming KG.
#  Proc Natl Acad Sci USA. 2011 Jun 21;108(25):10174-7.
#
#  More negative means more hydrophobic, values relative to alanine.
attribute: mfHydrophobicity

'''
hydro_index_d=[0,0.0,0.49,2.95,1.64,-2.2,1.72,4.76,-1.56,5.39,-1.81,-0.76,3.47,-1.52,3.01,3.71,1.83,1.78,\
                      -0.78,-0.38,-1.09]

###e.2006
'''
#  "An amino acid "transmembrane tendency" scale that approaches the
#   theoretical limit to accuracy for prediction of transmembrane helices:
#   relationship to biological hydrophobicity,"
#  Zhao and London, Protein Sci 15:8 (2006), doi: 10.1110/ps.062286306.
#
#  Zero should be at the middle of the extents (-3.46+1.98)/2 = -0.74
#
#  More positive means more hydrophobic.
attribute: ttHydrophobicity

'''
hydro_index_e=[0,0.38,-0.3,-3.27,-2.9,1.98,-0.19,-1.44,1.97,-3.46,1.82,1.4,-1.62,-1.44,-1.84,-2.57,-0.53,\
                      -0.32,1.46,1.53,0.49]


###pH=2: Normalized from Sereda et al., J. Chrom. 676: 139-153 (1994)
hydro_index_p2=[47,52,-18,8,92,0,-42,100,-37,100,74,-41,-46,-18,-26,-7,13,79,84,49]

###pH 7 values: Monera et al., J. Protein Sci. 1: 319-329 (1995).
hydro_index_p7=[41,49,-55,8,92,0,8,99,-23,97,74,-28,-46,-10,-14,-5,13,76,97,63]



