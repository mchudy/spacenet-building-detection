
import numpy as np

def GrowCluster ( intensity, cluster, mask, k ):
    newcluster = cluster
    #find boundary of current cluster
    indexes = np.nonzero( np.logical_and(mask, (np.logical_not(newcluster))))
    while indexes :
        growingboundary = 0 * cluster
        for kk in range ( len (indexes[0] ) ):
            i = indexes[0][kk]
            j = indexes[1][kk]
            isConnected = False
            isDecreasing = False
            if ( i > 0 ):
                if (newcluster[i-1,j] == k):
                    isConnected = True
                    if intensity[i-1,j] >= intensity[i,j]:
                        isDecreasing = True
            if ( i < newcluster.shape[0]-1 ):
                if (newcluster[i+1,j] == k):
                    isConnected = True
                    if intensity[i+1,j] >= intensity[i,j]:
                        isDecreasing = True
            if ( j > 0 ):
                if (newcluster[i,j-1] == k):
                    isConnected = True
                    if intensity[i,j-1] >= intensity[i,j]:
                        isDecreasing = True
            if ( j < newcluster.shape[1]-1 ):
                if (newcluster[i,j+1] == k):
                    isConnected = True
                    if intensity[i,j+1] >= intensity[i,j]:
                        isDecreasing = True
            if ( i > 0 ) and ( j > 0 ):
                if (newcluster[i-1,j-1] == k):
                    isConnected = True
                    if intensity[i-1,j-1] >= intensity[i,j]:
                        isDecreasing = True
            if ( i < newcluster.shape[0]-1 ) and (j > 0):
                if (newcluster[i+1,j-1] == k):
                    isConnected = True
                    if intensity[i+1,j-1] >= intensity[i,j]:
                        isDecreasing = True
            if ( j < newcluster.shape[1]-1 ) and  ( i < newcluster.shape[0]-1 ):
                if (newcluster[i+1,j+1] == k):
                    isConnected = True
                    if intensity[i+1,j+1] >= intensity[i,j]:
                        isDecreasing = True
            if isDecreasing and isConnected:
                growingboundary[i,j] = k
        if ( np.count_nonzero( growingboundary ) ):
            newcluster = newcluster + growingboundary
            indexes = np.nonzero( np.logical_and(mask, (np.logical_not(newcluster))))
        else:
            break
    return newcluster

def FindLargestCluster( intensity, mask, k ):
    peak = np.argmax(np.multiply( intensity, np.logical_and(intensity, mask)))
    peaki = peak//intensity.shape[1]
    peakj = peak - peaki * intensity.shape[1]
    cluster = 0 * intensity
    cluster[peaki,peakj] = k
    return GrowCluster( intensity, cluster, mask, k )


def FindAllClusters( intensity ):
    mask = (intensity >= 0.5).astype(int)
    cluster = 0 * intensity
    k = 1
    while (np.amax(mask) > 0 ):
        k = k + 1
        cluster += FindLargestCluster ( intensity, mask, k )
        mask = np.logical_and(mask, np.logical_not(cluster))
    return cluster
