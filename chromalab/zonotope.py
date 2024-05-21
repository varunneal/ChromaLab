import numpy as np
import math
import itertools
from tqdm import tqdm

# Heavily borrowed from Mathematica implementation written by Dave Eppstein
# https://ics.uci.edu/~eppstein/junkyard/ukraine/ukraine.html


# zSignTest ------------
# input: matrix 
# output: sign of hyperplane
# ----------------------
def zSignTest(matrix, facet_ids):
    extrabit = np.array([[(i + 200)**j for j in range(1, 1 + matrix.shape[0])] for i in range(1, matrix.shape[0] - len(facet_ids) + 1)])
    if extrabit.size == 0:
        return np.linalg.det(matrix.T[facet_ids, ::])
    else:
        return np.linalg.det(np.r_[matrix.T[facet_ids, ::], extrabit])

# getFacets ------------
# input: matrix and dimension of the facet - the matrix vector dim does not have to match the facet dim
# output: indices into matrix of the faces
# ----------------------
def getFacetIDs(matrix, dim, verbose=False): # matrix only used for the sign test
    num_elems = matrix.shape[1]
    list_facets = list(itertools.combinations(list(range(num_elems)), dim-1))

    valid_faces = set()
    faces_alr_looped = {}
    # first find all valid faces (groups of faces that lie on the same hyperplane)
    # find all valid faces such that if you add another vector to the face, it won't 
    # and the face doesn't exist already either (dictionary)
    for facet in tqdm(list_facets, disable= not verbose):
        # check if already in facet?
        if( facet in valid_faces or facet in faces_alr_looped): # 
            continue
        # check if adding the rest of the vectors lie on the face as well
        facetsInPlane = list(facet)
        ogfacetvecs = matrix.T[facet, ::]
        # accumfacetvecs = matrix.T[facet, ::]
        for i in range(num_elems):
            if i in facet: 
                continue
            else:
                det = zSignTest(matrix, list(facet) + [i])
                if math.isclose(det, 0, rel_tol=1e-5):
                    facetsInPlane +=[i]
                    # accumfacetvecs = np.r_[f'0,2',accumfacetvecs, matrix.T[i]]
        # add to dictionary & all of its two pair subspaces
        
        valid_faces.add(tuple(facetsInPlane))
        allfacets = list(itertools.combinations(facetsInPlane, 2))
        for i in range(len(allfacets)):
            faces_alr_looped[allfacets[i]] = True
    return valid_faces
        

# getFacetSums ------------
# input: list of facets, matrix and dimension of the facet - the matrix vector dim does not have to match the facet dim
# output: positive and negative sum at each facet
# -------------------------
def getFacetSums(facet_ids, matrix, dim, verbose=False):
    facetSums = {}
    # then for each face, separate the spectral locus and add up the vectors
    for facet_id in tqdm(facet_ids, disable=not verbose):
        # compute the normal vector to the plane
        positivesum = np.zeros(matrix.shape[0])
        negativesum = np.zeros(matrix.shape[0])
        for i,x in enumerate(matrix.T):
            if i in facet_id: 
                continue
            # sign = np.dot(normalvec.T, x)
            sign = zSignTest(matrix, list(facet_id)[:dim-1] + [i])
            if sign > 0:
                positivesum += x
            elif sign < 0:
                negativesum += x
            else:
                raise Exception("Something is on the face that wasn't accounted for")
        facetSums[facet_id] = [positivesum, negativesum]
    return facetSums

def getReflectance(cutpoint_ids, start, matrix, dim, maxval=1.0, verbose=False):
    # then for each face, separate the spectral locus and add up the vectors
    ref = np.zeros(matrix.shape[1])
    for i,x in enumerate(matrix.T):
        if i in cutpoint_ids: 
            continue # all faces are 0
        sign = zSignTest(matrix, list(cutpoint_ids)[:dim-1] + [i])
        if sign > 0: # changed to >=
            ref[i] = start
        elif sign < 0:
            ref[i] = 1-start
        else: # sign == 0, it's on the face, how to decide?
            raise Exception("Something is on the face that wasn't accounted for")
    return ref * maxval

# getFacet ------------
# input: list of facet ids representing the facet, matrix and dimension of the facet - the matrix vector dim does not have to match the facet dim
# output: return the facet in polygon ordering
# ---------------------
def getFacet(facet_ids, matrix, dim):
    if dim == 1:
        return np.zeros(matrix.shape[0])
    elif dim == 2 and len(facet_ids) == 1: # facet of 2D object is a line of origin and that vector
        return np.array([np.zeros(matrix.shape[0]), matrix.T[facet_ids]])
    else:
        a, b, zono = getZonotope(matrix.T[facet_ids, ::].T, dim-1)
        try:
            return np.unique(np.array(zono), axis=1).squeeze()
        except:
            return zono
        


# getZonotope ------------
# input: list of facets, matrix and dimension of the facet - the matrix vector dim does not have to match the facet dim
# output: positive and negative sum at each facet
# ------------------------
def getZonotope(matrix, dim, verbose=False):
    
    facetids = getFacetIDs(matrix, dim, verbose)
    facet_sums = getFacetSums(facetids, matrix, dim, verbose)

    posfaces = []
    negfaces = []
    
    for facet_ids in tqdm(facetids, disable=dim!=4):
        pos_offset, neg_offset = facet_sums[facet_ids]
        polygon = getFacet(facet_ids, matrix, dim)
        posfaces += [[x + pos_offset for x in polygon]]
        negfaces += [[x + neg_offset for x in polygon]]
    posfaces.extend(negfaces) # merge final into posfaces
    return facetids, facet_sums, posfaces

def getZonotopePoints(matrix, dim, verbose=False):
    facetids = getFacetIDs(matrix, dim, verbose)
    facet_sums = getFacetSums(facetids, matrix, dim, verbose)
    return [facetids, facet_sums]



def orderFace(face):
    ordered_list = [face[0]]
    for i in range(len(face)-1):
        cur_face = ordered_list[i]
        next_elem = None
        for j, next_cand in enumerate(face):
            if np.isclose(cur_face[1], next_cand[0], rtol=1e-08).all() and not np.isclose(cur_face[0], next_cand[1], rtol=1e-08).all():
                next_elem = next_cand
                break
            elif np.isclose(cur_face[1], next_cand[1], rtol=1e-08).all() and not np.isclose(cur_face[0], next_cand[0], rtol=1e-08).all():
                next_elem = next_cand[::-1, :]
                break
        if next_elem is None:
                ordered_list.append(ordered_list[-1])
        else: 
            ordered_list.append(next_elem)
    return ordered_list
    