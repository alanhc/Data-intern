import utils.pylib as py
from utils.SQL import postgres_connector, query_database
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# ML packetge
from sklearn.naive_bayes import GaussianNB
import numpy as np
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import io

class MLsolution():
    def __init__(self):
        self.utils = py.Utils()
        self.utils.get_args()
        self.phase = "error"
        self.host = self.utils.host
        self.port = self.utils.port
        self.model_path = self.utils.model_path
        self.save_path = self.utils.save_path
        
        if self.utils.filename=='test.py':
            self.phase='test'
            print('Load model from: ./save_models/'+self.model_path)
            self.model=load('./save_models/'+self.model_path)
            
        elif self.utils.filename=='train.py':
            self.phase='train'
        elif self.utils.filename=='evaluate.py':
            self.phase='evaluate'
            self.model=load('./save_models/'+self.model_path)

            

    
    def get_data(self):
        print('---------',self.phase)
        # connect to database
        print('Connect SQL on\n  host:'+str(self.host)+'\n  port:'+str(self.port))
        print('Getting data...')
        test_table_list=['post_shared', 'post_comment_created', 'post_liked', 'post_collected']
        evaluate_table_list=['posts','post_shared', 'post_comment_created', 'post_liked', 'post_collected'] #FOR TESTING!!!
        train_table_list=['posts','post_shared', 'post_comment_created', 'post_liked', 'post_collected']
        
        if self.phase=='train':
            table_list = train_table_list
        elif self.phase=='test':
            table_list = test_table_list
        elif self.phase=='evaluate':
            table_list= evaluate_table_list 
        else:
            table_list="known"
        
        data = pd.DataFrame()
        table_name=""
        for table in table_list:
            phase = self.phase
            if phase=='evaluate':
                phase='test'
            table_name=table+'_'+phase
            postgres_connector(host=self.host, port=self.port, database='intern_task', user='candidate', password=None)
            q1 = 'SELECT * FROM '+table_name
            r = query_database(q1,host=self.host, port=self.port)
            # rename count col
            r = r.rename(columns={'count':'count_'+table_name})
            # fomate created_at_hour 
            df = pd.to_datetime(r['created_at_hour'], errors='coerce')
            r['created_at_hour'] = df.dt.strftime('%Y-%m-%d %H:%M:%S')
            print('  '+table_name,':',r.shape)
            # append data and fill NaN
            data = data.append(r,ignore_index=True, sort=False)
            data = data.fillna(0)
        print(data.shape)
        # save clean data
        data.to_csv('output/'+self.phase+'_clean_data.csv',index=False)
    
    def preprocessing_data(self):
        print('Preprocessing data...')
        data = pd.read_csv('output/'+self.phase+'_clean_data.csv')
        print('  origin shape:', data.shape)
        # group history data
        g = data.groupby('post_key')
        max_n = g.size().max()
        min_n = g.size().min()
        print('  history range:('+str(min_n)+','+str(max_n)+')')
        # sum history data to make feature data
        df = g.sum()
        if self.phase=='test':
            self.features = df.loc[:, ~df.columns.isin(['like_count_36_hour','post_key','created_at_hour','count_post_comment_created_test'])]
            print('  features shape:', self.features.shape)
            #print(self.features.head(3))
        elif self.phase=='evaluate':
            self.features = df.loc[:, ~df.columns.isin(['like_count_36_hour','post_key','created_at_hour','count_post_comment_created_test'])]
            print('  features shape:', self.features.shape)
        elif self.phase=='train':
            # make answer 
            df['answer'] = pd.DataFrame(df['like_count_36_hour']>=1000)
            # make feature(X_train), answer(y_train)
            self.features = df.loc[:, ~df.columns.isin(['post_key','created_at_hour','answer','count_post_comment_created_train'])]
            self.targets = df['answer']
            print('  features shape:', self.features.shape)
            print('  targets shape:', self.targets.shape)

            # let X_train's rows count = y_train's rows count to avoid over-fitting   
            
            allTarget = self.targets.value_counts(sort=False)
            print("  y_train's shape")
            print(allTarget)
            # plot origin's each target label's sum
            ax = allTarget.plot(kind='bar',title='dataAll religion')
            ax.set_ylabel('number of class')
            ax.set_xlabel('class')
            ax.set_title('origin data')
            plt.savefig('output/img/origin_data.png')
            plt.cla()

            false_rows = df[df['answer']==False]
            true_rows = df[df['answer']==True]
            print('  false:'+str(false_rows.shape))
            print('  true:'+str(true_rows.shape))
            if false_rows.shape[0]>true_rows.shape[0]:
                bigger_rows = false_rows
                smaller_rows = true_rows
            else:
                bigger_rows = true_rows
                smaller_rows = false_rows

            # random sampling
            (tmp, bigger_rows) = train_test_split(false_rows, test_size=(smaller_rows.shape[0]/bigger_rows.shape[0]))
            
            print('  '+str(bigger_rows.shape))
            print('  '+str(smaller_rows.shape))
            
            
            df2 = bigger_rows
            df2 = df2.append(smaller_rows)
            
            self.features = df2.loc[:, ~df2.columns.isin(['like_count_36_hour','post_key','created_at_hour','answer','count_post_comment_created_train'])]
            self.targets = df2['answer']
            
            
            allTarget = self.targets.value_counts(sort=False)
            print("  y_train's shape")
            print(allTarget)
            # plot origin's each target label's sum
            ax = allTarget.plot(kind='bar',title='dataAll religion')
            ax.set_ylabel('number of class')
            ax.set_xlabel('class')
            ax.set_title('after_sampling data')
            plt.savefig('output/img/after_sampling_data.png')
            plt.cla()
            print('  done.')
            
            
            

    def run_predict(self, data):
        print('Predicting data...')
        print('  '+str(data.shape))
        
        y_predicts = self.model.predict(data)
        print('  y_predicts:'+str(y_predicts.shape))
        print('  done.')
        print('Saving data to output/'+self.save_path)
        answer = pd.DataFrame([self.features.index,y_predicts])
        answer = answer.T
        answer = answer.rename(columns={0:'post_key'})
        answer = answer.rename(columns={1:'is_trending'})
        answer['is_trending'] = answer['is_trending'].astype(bool).astype(int)
        print(answer.dtypes)
        answer.to_csv('output/'+self.save_path, index = False)
        print('  success to save answer! path: output/'+self.save_path)
        return y_predicts
    
    def run_train(self, X_train, y_train):
        print('Training data...')
        print('  X_train',X_train.shape)
        print('  y_train',y_train.shape)
        #print(X_train.columns)
        models_declare = [GaussianNB(), DecisionTreeClassifier(), SVC(),KNeighborsClassifier()]
        model_names = ['GaussianNB', 'DecisionTreeClassifier', 'SVC', 'KNeighborsClassifier']
        models=[]
        
        for i in range(len(models_declare)):
            print('  fitting model...'+model_names[i])
            model = models_declare[i]
            models.append(
                model.fit(X_train,y_train)
            )
            print('  saving model...'+model_names[i])
            dump(model,'save_models/'+model_names[i]+'.joblib') 
            print('  done.')
    def run_evaluate(self, data):
        file = open('output/record/'+self.model_path+'.txt', 'w+', encoding='UTF-8')
        
        evaluate = pd.read_csv('output/evaluate_clean_data.csv')
           
        evaluate['answer']=pd.DataFrame(evaluate['like_count_36_hour']>=1000)
        g = evaluate.groupby('post_key')
           
        max_n = g.size().max()
        min_n = g.size().min()
        df = g.sum()

        y_test = df['answer']

        ############################
        
        model = self.model
        
        y_predicts = model.predict(data)
        print(' y_predicts'+str(y_predicts.shape))
        print(' y_test'+str(y_test.shape))
        report = classification_report( y_predicts, y_test )
        print(report)
        file.write(str(report)+'\n')
        file.flush()
        mat = confusion_matrix(y_predicts, y_test)
        print(mat)
        file.write(str(mat)+'\n')
        file.flush()
        # caculate acc
        trace_sum = mat.trace()
        sum_all = np.sum(mat)
        print(str(trace_sum)+'/'+str(sum_all))
        print('acc:',trace_sum/sum_all)
        file.write('acc:'+str(trace_sum/sum_all)+'\n')
        file.flush()
        # plot 
        print(np.unique(y_test.values))
        target_names=np.unique(y_test.values)
        sns.heatmap(mat.T, square=True, annot=True, cbar=True, 
                xticklabels=target_names, yticklabels=target_names
        )
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.title(str(self.model_path))
        plt.savefig('output/img/'+self.model_path+'result_matrix.png')
        file.write('save image at:output/img/result_'+self.model_path+'_matrix.png\n')
        file.flush()
        plt.show(sns)
        plt.cla()
        file.close()


        

        
        
       
        
            



        



    


       
        
        




        
        
        
        
         
    
