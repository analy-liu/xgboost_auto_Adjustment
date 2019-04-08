import xgboost as xgb
from sklearn.model_selection import GridSearchCV

#cv select best num_boost_round(xg)
def modelfit(dtrain,params,useTrainCV = True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        cvresult = xgb.cv(params, dtrain, num_boost_round = params['n_estimators'], nfold=5,
         metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval = True)
        params['n_estimators'] = cvresult.shape[0]
    num_rounds = params['n_estimators']
    # train model
    model = xgb.train(params, dtrain, num_rounds)
    print(cvresult.shape[0])
    return model,num_rounds

#xgb_gsearch
def xgb_gsearch(X_train,y_train,num_rounds,F_T):
    #max_depth and min_child_weight
    param_test = {'max_depth' : range(5,7,1), #最大深度
             'min_child_weight':range(0,10,1) #决定最小叶子节点样本权重和
                 }
    estimator = xgb.XGBClassifier(                
      objective ='binary:logistic', #学习目标：二元分类的逻辑回归，输出概率
                               colsample_bytree = 0.9, #子采样率
                               eta = 0.3, #学习速率
                               n_estimators = num_rounds, #最大迭代次数
                               scale_pos_weight = F_T, #正负权重平衡
                               max_delta_step = 0, #子叶输出最大步长
                               subsample = 0.9 ,#训练实例的子样本比率
                               gamma = 1.6 ,#节点分裂所需的最小损失函数下降值
                               nthread = 4 ,#线程速率
                               reg_lambda = 0.1, #L2正则化速率
                               reg_alpha = 1e-5 ,#L1正则化速率
                               max_depth = 6, #最大深度
                               min_child_weight= 3 #决定最小叶子节点样本权重和
    )
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=5 )
    gsearch.fit(X_train,y_train)
                  
    #gamma
    param_test = {'gamma' : [i/10 for i in range(0,30,1)], #节点分裂所需的最小损失函数下降值
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
                  
    #colsample_bytree and subsample
    param_test = {'colsample_bytree' : [i/10.0 for i in range(6,10,1)], #子采样率
              'subsample':[i/10.0 for i in range(6,10,1)]#训练实例的子样本比率
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
    best_parameters = gsearch.best_estimator_.get_params()
    
    best_param = []
    for param_name in sorted(param_test.keys()):
        best_param.append([param_name, best_parameters[param_name]])
    p1 = best_param[0][1]
    p2 = best_param[1][1]
    max_edge1 = int(p1*100+5)
    min_edge1 = int(p1*100-5)
    max_edge2 = int(p2*100+5)
    min_edge2 = int(p2*100-5)               
    param_test = {'colsample_bytree' : [i/100.0 for i in range(min_edge1,max_edge1,1)], #子采样率
              'subsample':[i/100.0 for i in range(min_edge2,max_edge2,1)]#训练实例的子样本比率
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
                  
    #reg_alpha and reg_lambda
    param_test = {'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100], #L1正则化速率
              'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100], #L2正则化速率
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
                  
    #max_delta_step
    param_test = {'max_delta_step' : range(0,15)#子叶输出最大步长
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
    
    # eta
    param_test = {'eta' : [i/10.0 for i in range(1,10)],#学习速率
             }
    estimator = gsearch.best_estimator_
    gsearch = GridSearchCV(estimator,param_grid = param_test ,n_jobs = 4,scoring='roc_auc', cv=3 )
    gsearch.fit(X_train,y_train)
    return gsearch

def get_sk_params(gsearch):
    params = gsearch.best_estimator_.get_params()
    params_sk = {                
      'objective' :'binary:logistic', #学习目标：二元分类的逻辑回归，输出概率
                               'colsample_bytree' :0.55, #子采样率
                               'eta' : 0.3, #学习速率
                               'max_depth' : 9, #最大深度9
                               'n_estimators' : 1000, #最大迭代次数 748
                               'scale_pos_weight' : 1, #正负权重平衡
                               'max_delta_step' : 0, #子叶输出最大步长
                               'subsample' : 0.74 ,#训练实例的子样本比率
                               'gamma' : 0.0,#节点分裂所需的最小损失函数下降值
                               'min_child_weight' : 3 ,#决定最小叶子节点样本权重和
                               'nthread' : 4 ,#线程数
                               'alpha' : 1e-05 ,#L1正则化速率
                               'lambda' : 1e-05 #L2正则化速率
    }
    params["alpha"] = params.pop("reg_alpha")
    params["lambda"] = params.pop("reg_lambda")
    params_list = list(params)
    for param in params_list:
        if param not in params_sk:
            params.pop(param)
    return params

def set_params(gsearch,X_train,y_train,F_T):
    
    # parameter
    if gsearch == None:
        params = {                
          'objective' :'binary:logistic', #学习目标：二元分类的逻辑回归，输出概率
                               'colsample_bytree' :0.8, #子采样率
                               'eta' : 0.3, #学习速率
                               'max_depth' : 9, #最大深度9
                               'n_estimators' : 1000, #最大迭代次数 748
                               'scale_pos_weight' : 1, #正负权重平衡
                               'max_delta_step' : 0, #子叶输出最大步长
                               'subsample' : 0.8 ,#训练实例的子样本比率
                               'gamma' : 0.0,#节点分裂所需的最小损失函数下降值
                               'min_child_weight' : 3 ,#决定最小叶子节点样本权重和
                               'nthread' : 4 ,#线程数
                               'alpha' : 1e-05 ,#L1正则化速率
                               'lambda' : 1e-05 #L2正则化速率

        }
    else:
        params = get_sk_params(gsearch)
    # set model
    dtrain = xgb.DMatrix(X_train, y_train)
    # dtest = xgb.DMatrix(X_test, y_test)
    num_rounds = params['n_estimators']
    return params,dtrain,num_rounds