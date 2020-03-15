from FederatedCF import *
import json


file_path = '/home/jyfan/data/MoiveLens/ml-1m/'
loc_round = 10
attack_num = 6000

Test_Case = FederatedCF(file_path, attacker=attack_num, defense_alg=True)
Test_Case.set_loc_iter_round(loc_round)
Test_Case.set_lr(server_lr=0.1, client_lr=1e-4)
print("===== parameters =====")
print("E =", Test_Case.E)
print("Attackers = ", Test_Case.attacker)
print("Server lr = ", Test_Case.server_lr)
print("Client lr = ", Test_Case.client_lr[0])
print("Dim Feature = ", Test_Case.feature)
print("Penalty Factor = ", Test_Case.Lambda)
print("Defense Att = ", Test_Case.defense_alg)

#print("==== CF Model ====")
x = []  # round num
y = []  # loss

parameter = {'client_lr': Test_Case.client_lr,
             'server_lr': Test_Case.server_lr,
             'dim_feature': Test_Case.feature,
             'E': Test_Case.E,
             'penalty factor': Test_Case.Lambda}

init_loss = float(Test_Case.RMSE())
print(init_loss)
y.append(init_loss)

#Test_Case.save_global_model(0)
# Global iteration
for _ in range(100):
    Test_Case.global_update()
    avg_loss = Test_Case.RMSE()
    print('round', _+1, 'rmse (test case)=', float(avg_loss))
    y.append(float(avg_loss))
    #Test_Case.save_global_model(_+1)


data = {'RMSE': y, 'parameter': parameter}

with open('/home/jyfan/data/FLRS/FoolGlod_1m_attack'+str(attack_num)+'_fcf.json', 'w') as f:
#with open('/home/jyfan/data/FLRS/1m_attack'+str(attack_num)+'_fcf.json', 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
print("Finish.")
