from FederatedCF import *
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#file_path = '/home/jyfan/data/MoiveLens/ml-latest-small/'
file_path = '/home/jyfan/data/MoiveLens/ml-100k/'
#file_path = '/home/jyfan/data/MoiveLens/ml-1m/'
loc_round = 10


def train(attacker_par, detection):
    Test_Case = FederatedCF(file_path, attack_par=attacker_par, detection_alg=detection)
    Test_Case.set_loc_iter_round(loc_round)
    Test_Case.set_lr(server_lr=0.1, client_lr=1e-4)
    print("===== parameters =====")
    print("E =", Test_Case.E)
    print("Server lr = ", Test_Case.server_lr)
    print("Client lr = ", Test_Case.client_lr[0])
    print("Dim Feature = ", Test_Case.feature)
    print("Penalty Factor = ", Test_Case.Lambda)
    print("")
    print("Defense Att = ", Test_Case.detection_alg)
    print("Attack Model = ", Test_Case.attack_model)
    print("Target Item = ", Test_Case.target_item)
    print("Attackers = ", Test_Case.attacker_num)
    print("Fill items= ", Test_Case.fill_item_num)

    # x = []  # round num
    y1 = []  # loss
    y2 = []

    parameter = {'client_lr': Test_Case.client_lr,
                 'server_lr': Test_Case.server_lr,
                 'dim_feature': Test_Case.feature,
                 'E': Test_Case.E,
                 'penalty factor': Test_Case.Lambda}

    init_loss, global_loss = Test_Case.RMSE()
    print('init target rmse=', float(init_loss), 'init global rmse=', float(global_loss))
    y1.append(float(init_loss))
    y2.append(float(global_loss))

    # Global iteration
    for _ in range(200):
        Test_Case.global_update()
        target_loss, global_loss = Test_Case.RMSE()
        print('round', _+1, 'target rmse=', float(target_loss), 'golbal rmse=', float(global_loss))
        y1.append(float(target_loss))
        y2.append(float(global_loss))

    data = {'target rmse': y1, 'global rmse': y2, 'parameter': parameter}
    is_detection = ''
    if detection:
        is_detection = 'Detection'
    with open('/home/jyfan/data/FLRS/Shilling/' +
              is_detection + '_' +
              file_path.split('/')[-2] + '_' +
              attacker_par['model'] + '_target' + str(attacker_par['target']) +
              '_attack' + str(attacker_par['num']) + '_fill' + str(attacker_par['fill']) + '_fcf.json', 'w') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    #Test_Case.save_delta_item('/home/jyfan/data/FLRS/delta_item/small/target49/' + str(Test_Case.attacker_num) + 'attacker/')
    print("Finish.")


attacker_par = {
    'num': 583,
    'fill': 116,
    'model': 'random',
    'target': 49   # total 485, average score = 3.15
}
train(attacker_par, False)

attacker_par = {
    'num': 583,
    'fill': 116,
    'model': 'uniform',
    'target': 49   # total 485, average score = 3.15
}
train(attacker_par, False)










