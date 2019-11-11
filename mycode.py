import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel,delayed
import pandas as pd
class MultiClassSVMSGD(BaseEstimator,ClassifierMixin):
    def __init__(self, n_class=10,C=1,eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C=C
        self.type='ovr'
        self.n_class = n_class
        self.__estimators = []
        if(self.type=='ovr'):
            for i in range(n_class):
                self.__estimators.append(SVMSGD(C,eta,n_iter,random_state))
        else:
            for i in range(n_class):
                for j in range(i+1,n_class):
                    self.__estimators.append(SVMSGD(C,eta,n_iter,random_state))
    def get_params(self, deep=True):
        return {'C':self.C,'eta':self.eta}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        for est in self.__estimators:
            est.set_params(parameters=parameters)
                #est.setattr(parameter,value)
        return self
    def __ovr_inner_fit(self,clss,X,y):
        ny = np.array(y)
        ny[ny!=clss] = -1
        ny[ny==clss] = 1
        self.__estimators[clss].fit(X,ny)
        del ny
    def __inner_fit(self,clss1,clss2,X,y):
        neg = y==clss1
        pos = y==clss2
        nx = np.r_[X[neg],X[pos]]
        ny = np.r_[y[neg],y[pos]]
        ny[ny==clss1]=-1
        ny[ny==clss2]=1

        self.__estimators[clss].fit(X,ny)
        del ny
    def fit(self,X,y):
        if(self.type=='ovr'):
            result = Parallel(n_jobs=-1,require="sharedmem",verbose=True)(
                delayed(self.__ovr_inner_fit)(clss,X,y) for clss in range(self.n_class)
            )
        else:
            index = 0
            for i in range(self.n_class):
                for j in range(i+1,self.n_class):
                    neg = y==i
                    pos = y==j
                    nx = np.r_[X[neg],X[pos]]
                    ny = np.r_[y[neg],y[pos]]
                    ny[ny==i]=-1
                    ny[ny==j]=1
                    self.__estimators[index].fit(nx,ny)
                    index+=1


        """
        for clss in range(self.n_class):
            ny = np.array(y)
            ny[ny!=clss] = -1
            ny[ny==clss] = 1
            self.__estimators[clss].fit(X,ny)
        """
    def predict(self,X,y=None):
        rst = None
        if(self.type=='ovr'):
            def predict_inner(self,clss,X):
            
                return self.__estimators[clss].predict_prob(X)
            cls_probs = Parallel(n_jobs=-1,require="sharedmem",verbose=True)(delayed(predict_inner)(self,clss,X) for clss in range(self.n_class))#[]

            #for clss in range(self.n_class):
            #    cls_probs.append(self.__estimators[clss].predict_prob(X))
            cls_probs = np.array(cls_probs).T
            """
            print("cls_probs:",cls_probs.shape,cls_probs)
            sample = cls_probs[0]
            print("sample cls: ",np.argmin(sample),cls_probs.shape,list(sample))
            tmp = np.argmin(cls_probs,axis=1)
            for i in range(len(y)):
                if(tmp[i]!=y[i]):
                    print("wrong class"+str(i)+": ","(real="+str(y[i])+")",np.argmin(cls_probs[i]),list(cls_probs[i]))
                    print()
            del tmp
            """
            rst = np.argmin(cls_probs,axis=1)
            del cls_probs
        else:
            memo = np.zeros((len(X),self.n_class))
            index = 0
            for i in range(self.n_class):
                for j in range(i+1,self.n_class):
                    dec = self.__estimators[index].predict(X)
                    memo[dec<0,i]+=1
                    memo[dec>0,j]+=1
                    index+=1
            rst = np.argmin(memo,axis=1)
        return rst

class SVMSGD(BaseEstimator):
    
    def __init__(self, C=1,eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C=C
        self.__isW_Initialized = False
        self.__isB_Initialized = False
        self.__rgen = None
    def get_params(self, deep=True):
        return {'C':self.C,'eta':self.eta}
    def set_params(self, **parameters):
        print("call set_params from SVMSGD",parameters)
        self.C = parameters["parameters"]["C"]
        self.eta = parameters["parameters"]["eta"]
        #for parameter, value in parameters.items():
        #    setattr(self, parameter, value)
        print("\tC:",self.C,"eta:",self.eta)
        return self
    def __getRandomState(self):
        if(self.__rgen is None and (self.__isW_Initialized==False and self.__isB_Initialized == False)):
            self.__rgen = np.random.RandomState(self.random_state)
        return self.__rgen
    def __initW(self,m):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.wavg_ = self.w_.copy()
        self.__isW_Initialize = False
    def __initB(self,m):
        rgen = np.random.RandomState(self.random_state)
        self.b_ = rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.bavg_ = self.b_.copy()
        self.__isB_Initialize = False
    def __getPartOfCostFunc(self,xi,yi):
        return yi*(np.dot(xi,self.w_[1:].T)+self.b_[1:])
    def __getCond(self,xi,yi):
        precond = yi*(np.dot(self.w_[1:].T,xi))+self.b_[1:].mean() #self.__getPartOfCostFunc(xi,yi)
        #print(type(precond),precond.shape,precond)
        #print(sum(precond)/len(precond))
        return precond<=1
    def __updateBnW(self,mini_X,mini_y):
        mini_sz = len(mini_X)+0.0
        nw=0.0
        nb=0.0
        mini_costs = []
        for (xi,yi) in zip(mini_X,mini_y):
            if(self.__getCond(xi,yi)):
                nb-=yi
                nw-=yi*xi
            mini_costs.append(np.max(1-self.__getPartOfCostFunc(xi,yi),0).sum())
        b_errors = nb/mini_sz
        #print("b_errors:",type(b_errors))
        w1 = nw/mini_sz#len(mini_y)
        #print("w1:",type(w1),w1.shape)
        w2 = (self.C)*self.w_[1:]
        #print("w2:",type(w2),w2.shape)
        w_errors = w1+w2
        self.b_[1:] -= self.eta*b_errors
        self.b_[0]-=self.eta*b_errors
        self.w_[1:] -= self.eta*w_errors
        self.w_[0] -= self.eta*w_errors.mean()
        """
        print("updated weight:",self.w_)
        print("\t",self.w_[1:].sum(),self.w_[1:].mean())
        print("updated bias:",self.b_)
        print("\t",self.b_[1:].sum(),self.b_[1:].mean())
        """
        cost = sum(mini_costs)/mini_sz
        return cost
    def partial_fit(self,X,y):
        if(self.__isW_Initialized==False):
            self.__initW(X.shape[1])
        if(self.__isB_Initialized==False):
            self.__initB(X.shape[1])
        self.__updateWs(xi,tag)
        self.__updateBs(xi,tag)
        return self
    def fit(self, X, y):
        self.__isW_Initialized=False
        self.__isB_Initialized=False
        if(self.__isW_Initialized==False):
            self.__initW(X.shape[1])
        if(self.__isB_Initialized==False):
            self.__initB(X.shape[1])
        self.cost_ = []
        tot_len = len(X)
        l = tot_len//self.n_iter
        epoch = 1
        mg = None
        for ep in range(epoch):
            for i in range(1,self.n_iter+1):
                cost = []
                mini_X = X[(i-1)*l:(i)*l]
                mini_y = y[(i-1)*l:(i)*l]
                #for xi,tag in zip(X,y):
                    #self.__updateWs(xi,tag)
                cost.append(self.__updateBnW(mini_X,mini_y))
                self.wavg_[1:] = (i*self.wavg_[1:]+self.w_[1:])/(i+1)
                self.bavg_[1:] = (i*self.bavg_[1:]+self.b_[1:])/(i+1)
                if(mg is None):
                    self.wavg_use_ = self.wavg_.copy()
                    self.bavg_use_ = self.bavg_.copy()
                    mg = 0.5*(np.dot(self.wavg_[1:].T,self.wavg_[1:]))
                else:
                    nmg = 0.5*(np.dot(self.wavg_[1:].T,self.wavg_[1:]))
                    if(nmg<mg):
                        mg = nmg
                        self.wavg_use_ = self.wavg_.copy()
                        self.bavg_use_ = self.bavg_.copy()


                avg_cost = sum(cost)/len(mini_y)
                self.cost_.append(avg_cost)
        #print("wx: ",self.w_[1:])
        #print("w:",self.w_)
        #print("b:",self.b_)
        return self

    def net_input(self, X):
        tmp = np.dot(self.wavg_[1:],X.T)+self.bavg_[1:].mean()
        rst = np.array(tmp)
        return rst
    def calcMargin(self):
        return 2/(np.dot(self.w_[1:].T,self.w_[1:]))
    def activation(self, X):
        return self.net_input(X)
    def predict_prob(self,X):
        return np.dot(self.wavg_[1:].T,X.T)+self.bavg_[1:].mean()
    def predict(self, X):
        return  np.sign(self.activation(X))
