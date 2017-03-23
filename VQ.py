import ORB_Practice as OP
import scipy.cluster.vq as vq

def final_features(img, N):
    kp, des = OP.orb_descriptor(img)

    #print 'size of the descriptor prior to vector quantization: ', des.shape
    codebook = des
    #desWhiten = vq.whiten(des)

    #codebook, dist = vq.kmeans(desWhiten, N)

    #code, distort = vq.vq(desWhiten, codebook)

    #print 'size of the descriptor after vector quantization: ', codebook.shape

    return codebook


