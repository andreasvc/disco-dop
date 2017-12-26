# Please cite the following paper while re-using the code:
# Candito, M., Crabbé, B., & Denis, P. (2010, May).
# Statistical French dependency parsing: treebank conversion and first results.
# In Seventh International Conference on Language Resources and Evaluation-LREC 2010 (pp. 1840-1847).
# European Language Resources Association (ELRA).

# Copyright: Candito, M., Crabbé, B., & Denis, P.

# regex expressions for repeated compound patterns which will be undone
RegularCompoundPatterns = {# a N, maybe with Det, Adj, and PPs
                           # " Organisation_de_coopération_et_de_développement_économique "
                           # " Institut_de_formation_des_agents_de_voyages"
                           # " pomme de terre "
                           # "marché monétaire et obligataire"
                           # "	Bureau_de_recherches_géologiques_et_minières"
                           # attention : apres un P, on impose un N qqpart (le code s'appuie là-dessus...)
                           'N': ['((D )?(A )*N( A( C A)?)*( P(\+D)?( D)?( A)* N( A( C A)?)*(( C)? P(\+D)?( D)?( A)* N)*)?)'],
                           'V': ['(V )+(P|A|D)*( N)+( P)* | \
                                 (N|D)( P)*( N)+'], #faire face, faire appel
                           'P': ['P(\+D)? (D )?(A )*N( P(\+D)?( D)?( A)* N( A( C A)?)*(( C)? P(\+D)?( D)?( A)* N( A)*)*)? P(\+D)?'],
                           'ADV': ['P(\+D)? (D )?(A )*N( P(\+D)?( D)?( A)* N( A)*)?']}

# The compounds found in the AllowedCompound list should not be undone,
# those are mostly organization names or fixed expressions like 'aujourd'hui'
# TODO! extend list of allowed compounds

AllowedCompounds = [
                    '(MWN (N Fondation) (N France) (A active))',
                    '(MWN (N Côte) (PONCT -) (P d\') (N Ivoire))',
                    '(MWP (CL Il) (CL y) (V a))',
                    '(MWN (N Jean) (PONCT -) (N Louis))',
                    '(MWADV (ADV tout) (P de) (N suite))',
                    '(MWP (P jusqu\') (P au))',
                    '(MWN (A Haute) (PONCT -) (N Corse))',
                    '(MWP (CL y) (A compris))',
                    '(MWN (N Chalon) (PONCT -) (P sur) (PONCT -) (N Saône))',
                    '(MWP (CL il) (CL y) (V a))',
                    '(MWN (N Royaume) (PONCT -) (A uni))',
                    '(MWN (N Seine) (PONCT -) (N Saint) (PONCT -) (N Denis))',
                    '(MWN (N Roche) (PONCT -) (N la) (PONCT -) (N Molière))',
                    '(MWN (N Air) (N France))',
                    '(MWN (A Grande) (PONCT -) (N Bretagne))',
                    '(MWN (N Union) (A soviétique))'
                    ]