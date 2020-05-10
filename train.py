import utils.method as m


# python train.py best.joblib 35.187.144.113 5432 
def main():
    solution = m.MLsolution()
    # handle data
    solution.get_data()
    solution.preprocessing_data()
    #　make a X_train(feature), y_train for model's input 
    X_train = solution.features
    y_train = solution.targets

    solution.run_train(X_train,y_train)
    
        
if __name__ == "__main__":
    main()