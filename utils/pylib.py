import sys

# python test.py best.joblib 35.187.144.113 5432 answer.csv
params_format = {
    'test':'python test.py best.joblib 35.187.144.113 5432 answer.csv',
    'train':'python train.py best.joblib 35.187.144.113 5432',
    'evaluate':'python evaluate.py best.joblib 35.187.144.113 5432 answer.csv'
}
params_n = {
    'test':5,
    'train':4,
     'evaluate':5
}
class Utils():
    def __init__(self):
        self.host=""
        self.port=""
        self.model_path=""
        self.save_path=""
        args = sys.argv
        if args[0]=='test.py': 
            self.phase='test'
        elif args[0]=='train.py':
            self.phase='train'
        elif args[0]=='evaluate.py':
            self.phase='evaluate'

    def get_args(self):
        args = sys.argv
        #print('mode:', args)
        self.filename = args[0]
        if len(args)!=params_n[self.phase]:
            print('error parms, please try:', params_format[self.phase] )
            sys.exit()
        else:
            self.model_path = args[1]
            if self.phase=='test':
                self.save_path=args[4]
            self.host = args[2]
            self.port = int(args[3])
       
        
    
    
    
    
    
    