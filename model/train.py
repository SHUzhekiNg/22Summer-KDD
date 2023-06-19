from channelwise_GRU import *

ob = Metesre()

checkpoint_path = './model_checkpoint.h5'
ob.model.load_weights(checkpoint_path)

# recommendation = ob._predictor(
#         [ob.X1_test[:3000], ob.X2_test[:3000]])
# gnd = ob.y_test[:3000].tolist()
# test1_MRR = ob._mean_reciprocal_rank(recommendation, gnd)
# print(f'MRR for testset: {test1_MRR}')
# accuracy_score = ob._accuracy(recommendation, gnd)
# print(f'Acc for testset: {accuracy_score}')

for i in range(5):
    ob.train(epoch=1)
    recommendation = ob._predictor(
        [ob.X1_test[:3000], ob.X2_test[:3000]])
    gnd = ob.y_test[:3000].tolist()
    test1_MRR = ob._mean_reciprocal_rank(recommendation, gnd)
    print(f'MRR for testset: {test1_MRR}')
    accuracy_score = ob._accuracy(recommendation, gnd)
    print(f'Acc for testset: {accuracy_score}')


# test
# ob.test_1_testontest()
# ob.test_2_testwithendone()
# ob.test_3_generatefinalresult()
