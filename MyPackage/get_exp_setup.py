# this function generates the experimental setup files, which includes:
# # training / test data, learning rate, NN structure, bus number
# #of training epoches, batch size
import MyPackage.config as config
def get_exp_setup():
    with open(config.result_path + '/ExpSetup.txt', 'w') as f:  # 设置文件对象
        f.write('Experiment Setup:   '+ '\n')  # DNN structure
        f.write('BUS number:   ' + str (config.Nbus)+ '\n')  # DNN structure
        f.write('training/test data:   ' + str(config.Ntrain) + '/' + str(config.Ntest)+ '\n')  # training/test data
        f.write('Learning rate:   ' +  str(config.Lr)+ '\n')  # Learning rate
        f.write('Training epoch:   '+ str(config.Epoch)+ '\n')  # Training epoch
        f.write('Training batch size:   '+ str(config.batch_size_training)+ '\n')  # Training epoch
        f.write('DNN structure:   ' + str(config.hidden_units * config.khidden)+ '\n')  # DNN structure
        f.write('MSE weights:   ' + str(config.mse_weight) + '\n')  # MSE weights
        f.write('MSE weights:   ' + str(config.mse_weight) + '\n')  # MSE weights
