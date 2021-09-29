"""
Utilities related to computing SVDs
The SVD class is used within application.tasks.write_r_extracts, in combination with corpusvectorspace and tagvectorspace
"""

import os
import pickle as cPickle
import math
from loguru import logger
# from config import paths
import src.data.paths as paths
#from simcalc.sc_lib import utils
from src.data import utils

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

import sparsesvd

import pdb

DEFAULT_DIRECTORY = paths.DIR_LSI
MATLAB_INFILE = 'matlab_infile%s'
MATLAB_OUTFILE_SVD = 'matlab_outfile_svd%s.txt'
MATLAB_OUTFILE_TRANSFORMED_VECTORS = 'matlab_outfile_transformed_vectors%s.txt'
MATLAB_PROGRAM_NAME = 'doSVD%s'
MATLAB_EXTENSION = '.m'

USE_MATLAB = False

class SVD:

    def __init__(self, vectorSpace, svdRank=200, directory=DEFAULT_DIRECTORY, name=""):
        self.vectorSpace = vectorSpace
        self.directory = directory
        self.svdRank = svdRank
        self.name = name

    def setSvdRank(self, svdRank):
        self.svdRank = svdRank

    def generateMatlabProgram(self):
        """
        Write the Matlab code for generating the outfiles from the infile
        """
        infile_name = self.directory + '/' + (MATLAB_INFILE % self.name)
        svd_outfile_name = self.directory + '/' + (MATLAB_OUTFILE_SVD % self.name)
        vec_outfile_name = self.directory + '/' + (MATLAB_OUTFILE_TRANSFORMED_VECTORS % self.name)
        programSource = "load %s\n" \
                        "X = (spconvert(%s))';\n" \
                        "[U,S,V] = svds(X,%d);\n" \
                        "dlmwrite('%s',inv(S)*U')\n" \
                        "dlmwrite('%s',(inv(S)*U'*X)')\n" \
                        "exit" \
                        % (infile_name, MATLAB_INFILE % self.name, self.svdRank, svd_outfile_name, vec_outfile_name)
        f = open((MATLAB_PROGRAM_NAME % self.name) + MATLAB_EXTENSION, 'w')
        f.write(programSource)
        f.close()

    def runMatlabProgram(self):
        """
        Run the generated code on Matlab in a separate process
        """
        # p = Popen("matlab -r " + MATLAB_PROGRAM_NAME, shell=True)
        # p.wait()
        print('Beginning to run matlab program\n')
        os.system(paths.MATLAB_PATH + " -r " + (MATLAB_PROGRAM_NAME % self.name))

    def writeMatlab(self):
        """
        Create the data file used as input to the Matlab program
        """
        print('Beginning to write extract file for matlab\n')
        f = open(self.directory + '/' + (MATLAB_INFILE % self.name), 'w')
        for vectorIndex, vector in enumerate(self.vectorSpace.getVectors()):
            for featureIndex, featureValue in vector.iteritems():
                f.write(str(vectorIndex + 1).rjust(7) + str(featureIndex + 1).rjust(7) + ' %5.6f\n' % featureValue)
        f.close()

    def readMatlab(self):
        """
        Read the output files created by Matlab, and load the data into this object
        :return:
        """
        print('Beginning to read files created by matlab program\n')
        f = open(self.directory + '/' + (MATLAB_OUTFILE_SVD % self.name), 'r')
        self.svd = []
        for line in f:
            self.svd.append([float(val) for val in line.split(',')])
        f.close()
        f = open(self.directory + '/' + (MATLAB_OUTFILE_TRANSFORMED_VECTORS % self.name), 'r')
        self.transformedVectors = []
        for line in f:
            self.transformedVectors.append([float(val) for val in line.split(',')])
        f.close()

    def load(self):
        """
        Prepare for and run the Matlab program, and read the data from the outfiles into this object
        """
        if USE_MATLAB:
            self.writeMatlab()
            self.generateMatlabProgram()
            self.runMatlabProgram()
            self.readMatlab()
        else:
            self.load_without_matlab()


    def getTransformedVector(self, vectorIndex):
        return self.transformedVectors[vectorIndex]

    def getTransformedVectorByName(self, vectorName):
        return self.getTransformedVector(self.vectorSpace.getVectorIndex(vectorName))

    def getSimByNameAndVector(self, vectorName, vector2):
        transformedVector1 = self.getTransformedVectorByName(vectorName)
        transformedVector2 = self.transformVector(vector2)
        return self.getSim(transformedVector1, transformedVector2)

    def getSim(self, transformedVector1, transformedVector2):
        return utils.cosSim(transformedVector1, transformedVector2)

    def transformVector(self, vector):
        transformedVector = []
        for row in self.svd:
            transformedVector.append(sum([row[featureIndex] * featureVal for featureIndex, featureVal in vector.items()]))
        return transformedVector

    def reportVectorSim(self, vector):
        data = []
        transformedVector = self.transformVector(vector)
        for vectorIndex, vector in enumerate(self.vectorSpace.getVectors()):
            transformedVectorCompare = self.getTransformedVector(vectorIndex)
            sim = self.getSim(transformedVector, transformedVectorCompare)
            data.append((sim, self.vectorSpace.getVectorDesc(vectorIndex)))
        data.sort()
        data.reverse()
        for sim, vectorDesc in data:
            print(sim, vectorDesc)

    def writeToFile(self, fileName='svd'):
        cPickle.dump(self, open(paths.DIR_LSI + '/' + fileName + '.pickle', 'wb'))


    def load_without_matlab(self):
        """
        Do the load process without Matlab (doesn't create temporary outfiles or infiles either)
        """

        def build_matrix():
            """
            Load the data (in Matlab, it was loaded from the INFILE)
            """
            row_inds = []
            col_inds = []
            data = []
            for vectorIndex, vector in enumerate(self.vectorSpace.getVectors()):
                for featureIndex, featureValue in vector.items():
                    row_inds.append(vectorIndex)
                    col_inds.append(featureIndex)
                    data.append(featureValue)
            row_inds = scipy.array(row_inds)
            col_inds = scipy.array(col_inds)
            data = scipy.array(data).astype(np.float64)

            # Create a sparse matrix (In Matlab it was `X = (spconvert(INFILE))' `)
            matr = scipy.sparse.csc_matrix((data, (row_inds, col_inds))).T
            #try:
            #    # Create a sparse matrix (In Matlab it was `X = (spconvert(INFILE))' `)
            #    matr = scipy.sparse.csc_matrix((data, (row_inds, col_inds))).T
            #except ValueError:
            #    pdb.set_trace()
            return matr


        def determine_k(matr):
            """
            Make sure k isn't too big (only a problem for small datasets, Matlab did this automatically)
            :return: the largest acceptable value of k
            """
            return min(self.svdRank, min(matr.shape))  # Matlab did this by default

        def do_svd(matr, k):
            # TODO: Get sparse svd to work: `(u, s, vt) = scipy.sparse.linalg.svds(matr,k=k,tol=1e-10/math.sqrt(2),maxiter=300)`
            """
            Do SVD
                get the k largest eigenvalues "s" and their corresponding eigenvectors "u"
                where  u * diag(s) * v  approximates the matrix
                (In Matlab, this used to be `[U,S,V] = svds(X,%d);`)
            :param matr: (m x n)
            :return: ut, s
                s (k x k) has the largest eigenvalues along its diagonal (decreasing down the diagonal)
                ut (k x m) is the transpose of the matrix of eigenvectors, in order corresponding to s
            """
            # pdb.set_trace()
            ut, s, vt = sparsesvd.sparsesvd(matr.tocsc(), k)
            s = scipy.diag(s)

            # u, s, vt = scipy.sparse.linalg.svds(matr, k=k, tol=1e-10/math.sqrt(2),maxiter=300)
            # ut = u.T

            # # Do the SVD
            # (u, s, vt) = scipy.linalg.svd(matr, full_matrices=False)
            #
            # # Determine the order
            # ordered_indices = np.argsort(s)[::-1]  # `[::-1]` reverses the list
            # k_ordered_inds = ordered_indices[0:k]
            # s = np.array(s)[k_ordered_inds]
            # ut = np.array(u.T)[k_ordered_inds]
            #
            # # Put the eigenvalues on the diagonal
            # s = scipy.diag(s)

            return ut, s

        def calculate_svd_output_data(ut, s):
            """
            Prepare the svd output data
            (Matlab wrote it to a file `dlmwrite(outfile_svd, inv(S) * U')`)
            """
            return np.dot(scipy.linalg.inv(s), ut)

        def calculate_transformed_vector_output_data(svd_output_data, matr):
            """
            Prepare the transformed vector output data
            (Matlab wrote it to a file `dlmwrite(outfile_transformed_vectors, (inv(S)*U'*X)')`)
            """
            # take advantage of sparse matrix .dot():   (C = A*B)  === (C.T = B.T * A.T)
            return matr.T.dot(svd_output_data.T)

        def write_svd_outfile(svd_out):
            """
            Write the svd output data to a file (Matlab used dlmwrite)
            """
            with open(self.directory + '/' + (MATLAB_OUTFILE_SVD % self.name), 'w+') as f:
                for inner_list in svd_out:
                    f.write(','.join([str(val) for val in inner_list]) + '\n')

        def write_transformed_vector_outfile(transformed_vectors_out):
            """
            Write the transformed vector output data to a file (Matlab used dlmwrite)
            """
            with open(self.directory + '/' + (MATLAB_OUTFILE_TRANSFORMED_VECTORS % self.name), 'w+') as f:
                for inner_list in transformed_vectors_out:
                    f.write(','.join([str(val) for val in inner_list]) + '\n')

        def load_data_into_object(svd_out, transformed_vectors_out):
            """
            Make this object hold the data, preparing for the next step
            """
            self.svd = []
            for inner_list in svd_out:
                self.svd.append([float(val) for val in inner_list])
            self.transformedVectors = []
            for inner_list in transformed_vectors_out:
                self.transformedVectors.append([float(val) for val in inner_list])

        X = build_matrix()
        X_arr = X

        k = determine_k(X_arr)
        print("k:\t", k)
        ut, s = do_svd(X_arr, k)

        svd_out = calculate_svd_output_data(ut, s)
        transformed_vectors_out = calculate_transformed_vector_output_data(svd_out, X_arr)

        write_svd_outfile(svd_out)
        write_transformed_vector_outfile(transformed_vectors_out)

        load_data_into_object(svd_out, transformed_vectors_out)



def readFromFile(fileName='svd'):
    return cPickle.load(open(paths.DIR_LSI + '/' + fileName + '.pickle', 'rb'))

# Methods for serializing and testing objects from this class


def testSvdTag():
    from lsi import svd
    svd = svd.readFromFile()
    vector = svd.vectorSpace.makeVectorBinary(['western'])
    svd.reportVectorSim(vector)

def writeSvdCorpus():
    from lsi import svd
    import corpusvectorspace
    from corpus import imdbcorpus
    imdbCorpus = imdbcorpus.read(fileName='imdb_mindocs_100')
    corpusVectorSpace = corpusvectorspace.CorpusVectorSpace(imdbCorpus)
    svd = svd.SVD(corpusVectorSpace)
    svd.load()
    svd.writeToFile('svd_imdb_corpus')

def testSvdCorpus():
    from lsi import svd
    svd = svd.readFromFile('svd_imdb_corpus')
    for testText in ['cute', 'western']:
        vector = svd.vectorSpace.makeVector(testText)
        print(testText)
        svd.reportVectorSim(vector)


if __name__ == '__main__':
    writeSvdCorpus()
    testSvdCorpus()
