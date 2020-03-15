from CollaborativeFiltering import *
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#file_path = '/home/jyfan/data/MoiveLens/ml-latest-small/'
file_path = '/home/jyfan/data/MoiveLens/ml-1m/'

Test_Case = CollaborativeFiltering(file_path)

parameter = {'lr': Test_Case.lr,
             'dim_feature': Test_Case.feature,
             'penalty factor': Test_Case.Lambda}

y = []

init_rmse = float(Test_Case.RMSE())
print(init_rmse)
y.append(init_rmse)
print("--- start training ---")
#Test_Case.save_model(0)
# train model
for _ in range(100):
    Test_Case.train()
    rmse = Test_Case.RMSE()
    y.append(float(rmse))
    print("round ", _+1, " rmse:", y[-1])
#    Test_Case.save_model(_+1)

data = {'RMSE': y, 'parameter': parameter}
#with open('/home/jyfan/RS_FL/CF_baseline_100round.json', 'w') as f:
#    f.write(json.dumps(data, ensure_ascii=False, indent=2))
print("Finish.")
