import numpy as np

def decomp(mat):

    mat = list(mat)
    sym = []#np.empty(mat.shape[-2:])
    
    for m in mat:

        mat_flip_x = np.fliplr(m)

        mat_flip_y = np.flipud(m)
        mat_flip_xy = np.fliplr(np.flipud(m))
        sum = m + mat_flip_x + mat_flip_y + mat_flip_xy

        mat_sum_rot_90 = np.rot90(sum)
        #print((sum + mat_sum_rot_90) / 8)
        #sym = (sum + mat_sum_rot_90) / 8
        #anti_sym = mat - sym
        sym.append(((sum + mat_sum_rot_90) / 8))

    return np.array(sym) #, anti_sym


if __name__ == "__main__":

    mat = np.array([[[0.369362353	,0.686498	,0.533563106	,0.68618978,	0.810169907,	0.992890307,	0.753928141],
    [0.895605062	,0.858500596	,0.330535171	,0.34825479	,0.35083694	,0.861523825	,0.437993638],
    [0.645282398	,0.170230225	,0.133976179	,0.936895212	,0.919530674	,0.681907143,	0.489057997],
    [0.485997824	,0.770828642	,0.45839399	,0.274668289	,0.419419446	,0.060458453,	0.372236255],
    [0.33173995	,0.620286941	,0.954353626	,0.651997819	,0.567057449	,0.952286994	,0.674117601],
    [0.742861568	,0.390033226	,0.824093382	,0.45231723	,0.09097642	,0.381456982	,0.965336184],
    [0.789878595	,0.271517016	,0.322960481	,0.460666519,	0.959389661	,0.365591294	,0.692346819]]])
    #mat = [np.arange(7*7).reshape([7,7])]
    #print(mat)
    sym= decomp(mat)

    print(sym)