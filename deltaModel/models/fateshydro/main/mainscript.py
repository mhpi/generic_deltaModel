from Init.InitMod                  import InitMod
from models.train_test             import train_Model, test_Model, forward_model_single, forward_Model_batch
from data.dataloader   import dataloader
from conf.config_utils import set_config_args

#========================================================
args       = set_config_args()
dimsDict   = args['dimsDict']
dataDict   = dataloader(args).dataDict
InitMod    = InitMod(args, dataDict)
# train_Model(InitMod.args, InitMod.data, InitMod.models_Dict)
test_Model(InitMod.args, InitMod.data, InitMod.models_Dict)
#forward_model_single(InitMod.args, InitMod.data, InitMod.models_Dict)

# forward_Model_batch(InitMod.args, InitMod.data, InitMod.models_Dict)

