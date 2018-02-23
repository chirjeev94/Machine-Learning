import numpy as np
import scipy.io as spio
import scipy.stats as stats
import math

class GaussianProbabilisticModel:
    
    def __init__(self,num_classes,mle_mu_map,mle_sigma_map,class_prob_map):
        self.num_classes=num_classes
        self.mle_mu_map={}
        self.mle_sigma_map={}
        self.class_prob_map={}

    ##should be private
    def gaussian_pdf_helper(self,x,mu,sigma):
        rv=stats.multivariate_normal(mean=mu,cov=sigma+0.1*np.eye(len(mu),len(mu)))
        print(rv.pdf(x))
        return rv.pdf(x)
                
        ##  calculate f(x)~multivariate Normal density
    def train_model(self,training_data,labels):
        nums={}
        for i in range(len(training_data)):
            if (not (labels[i] in self.mle_mu_map)):
                self.mle_mu_map[labels[i]]=training_data[i]
                nums[labels[i]]=1
            else:
                print("waah")
                self.mle_mu_map[labels[i]]=self.mle_mu_map[labels[i]]+training_data[i]
                nums[labels[i]]=nums[labels[i]]+1
        
        for i in range(self.num_classes):
            self.mle_mu_map[i]=(1.0/nums[i])*self.mle_mu_map[i]
        
        for i in range(len(training_data)):
            if (not (labels[i] in self.mle_sigma_map)):
                self.mle_sigma_map[labels[i]]=(np.matrix(training_data[i]-self.mle_mu_map[labels[i]]).T)*(np.matrix(training_data[i]-self.mle_mu_map[labels[i]]))
                
            else:
                print("arrey")
                self.mle_sigma_map[labels[i]]=self.mle_sigma_map[labels[i]]+(np.matrix(training_data[i]-self.mle_mu_map[labels[i]]).T)*(np.matrix(training_data[i]-self.mle_mu_map[labels[i]]))
                       
        
        for i in range(self.num_classes):
            self.mle_sigma_map[i]=(1.0/nums[i])*self.mle_sigma_map[i]
            print(nums[i],len(training_data))
            self.class_prob_map[i]=(nums[i]*1.0)/len(training_data)*1.0
        ##Steps:
        ##1: Partition data according to labels
        
        ##2: Compute MLE of mu and sigma for each label
    def predict(self,x):
        prediction=0
        curr_max=0;
        for i in range(self.num_classes):
            p=self.gaussian_pdf_helper(x,self.mle_mu_map[i],self.mle_sigma_map[i])*self.class_prob_map[i]
            print(self.class_prob_map[i])
            if(p>curr_max):
                curr_max=p
                prediction=i
        return prediction
            
                
                
                
                
        
    def testing_error(self,test_data,labels):
        print()    
    
        ##Steps:
        ##1: return argmax P[X=x/Y=y]*P[Y=y]

mat = spio.loadmat('hw1data.mat', squeeze_me=True)
image_matrix=mat['X']
label_array=mat['Y']   

    
gaussian_model= GaussianProbabilisticModel(10,{},{},{})
gaussian_model.train_model(image_matrix[0:30,:],label_array[0:30])
#print(gaussian_model.mle_mu_map)
gaussian_model.train_model(image_matrix[1],label_array[1])
print("start")
