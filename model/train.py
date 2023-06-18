from channelwise_GRU import *

ob = Metesre()
for i in range(6):
    ob.train(epoch=1, from_checkpoint=True)
    n = 1 + i
    recommendation = ob._predictor(
        [ob.X1_test[:3000], ob.X2_test[:3000], ob.X3_test[:3000], ob.X4_test[:3000], ob.X5_test[:3000],
         ob.X6_test[:3000]])
    gnd = ob.y_test[:3000].tolist()
    test1_MRR = ob._mean_reciprocal_rank(recommendation, gnd)
    print(f'MRR for testset: {test1_MRR}')
